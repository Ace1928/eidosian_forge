import itertools
import logging
from typing import BinaryIO, Container, Dict, Iterator, List, Optional, Tuple
from pdfminer.utils import Rect
from . import settings
from .pdfdocument import PDFDocument, PDFTextExtractionNotAllowed, PDFNoPageLabels
from .pdfparser import PDFParser
from .pdftypes import PDFObjectNotFound
from .pdftypes import dict_value
from .pdftypes import int_value
from .pdftypes import list_value
from .pdftypes import resolve1
from .psparser import LIT
@classmethod
def create_pages(cls, document: PDFDocument) -> Iterator['PDFPage']:

    def search(obj: object, parent: Dict[str, object]) -> Iterator[Tuple[int, Dict[object, Dict[object, object]]]]:
        if isinstance(obj, int):
            objid = obj
            tree = dict_value(document.getobj(objid)).copy()
        else:
            objid = obj.objid
            tree = dict_value(obj).copy()
        for k, v in parent.items():
            if k in cls.INHERITABLE_ATTRS and k not in tree:
                tree[k] = v
        tree_type = tree.get('Type')
        if tree_type is None and (not settings.STRICT):
            tree_type = tree.get('type')
        if tree_type is LITERAL_PAGES and 'Kids' in tree:
            log.debug('Pages: Kids=%r', tree['Kids'])
            for c in list_value(tree['Kids']):
                yield from search(c, tree)
        elif tree_type is LITERAL_PAGE:
            log.debug('Page: %r', tree)
            yield (objid, tree)
    try:
        page_labels: Iterator[Optional[str]] = document.get_page_labels()
    except PDFNoPageLabels:
        page_labels = itertools.repeat(None)
    pages = False
    if 'Pages' in document.catalog:
        objects = search(document.catalog['Pages'], document.catalog)
        for objid, tree in objects:
            yield cls(document, objid, tree, next(page_labels))
            pages = True
    if not pages:
        for xref in document.xrefs:
            for objid in xref.get_objids():
                try:
                    obj = document.getobj(objid)
                    if isinstance(obj, dict) and obj.get('Type') is LITERAL_PAGE:
                        yield cls(document, objid, obj, next(page_labels))
                except PDFObjectNotFound:
                    pass
    return