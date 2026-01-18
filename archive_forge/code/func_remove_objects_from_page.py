import codecs
import collections
import decimal
import enum
import hashlib
import re
import uuid
from io import BytesIO, FileIO, IOBase
from pathlib import Path
from types import TracebackType
from typing import (
from ._cmap import build_char_map_from_dict
from ._doc_common import PdfDocCommon
from ._encryption import EncryptAlgorithm, Encryption
from ._page import PageObject
from ._page_labels import nums_clear_range, nums_insert, nums_next
from ._reader import PdfReader
from ._utils import (
from .constants import AnnotationDictionaryAttributes as AA
from .constants import CatalogAttributes as CA
from .constants import (
from .constants import CatalogDictionary as CD
from .constants import Core as CO
from .constants import (
from .constants import PageAttributes as PG
from .constants import PagesAttributes as PA
from .constants import TrailerKeys as TK
from .errors import PyPdfError
from .generic import (
from .pagerange import PageRange, PageRangeSpec
from .types import (
from .xmp import XmpInformation
def remove_objects_from_page(self, page: Union[PageObject, DictionaryObject], to_delete: Union[ObjectDeletionFlag, Iterable[ObjectDeletionFlag]]) -> None:
    """
        Remove objects specified by ``to_delete`` from the given page.

        Args:
            page: Page object to clean up.
            to_delete: Objects to be deleted; can be a ``ObjectDeletionFlag``
                or a list of ObjectDeletionFlag
        """
    if isinstance(to_delete, (list, tuple)):
        for to_d in to_delete:
            self.remove_objects_from_page(page, to_d)
        return
    assert isinstance(to_delete, ObjectDeletionFlag)
    if to_delete & ObjectDeletionFlag.LINKS:
        return self._remove_annots_from_page(page, ('/Link',))
    if to_delete & ObjectDeletionFlag.ATTACHMENTS:
        return self._remove_annots_from_page(page, ('/FileAttachment', '/Sound', '/Movie', '/Screen'))
    if to_delete & ObjectDeletionFlag.OBJECTS_3D:
        return self._remove_annots_from_page(page, ('/3D',))
    if to_delete & ObjectDeletionFlag.ALL_ANNOTATIONS:
        return self._remove_annots_from_page(page, None)
    jump_operators = []
    if to_delete & ObjectDeletionFlag.DRAWING_IMAGES:
        jump_operators = [b'w', b'J', b'j', b'M', b'd', b'i'] + [b'W', b'W*'] + [b'b', b'b*', b'B', b'B*', b'S', b's', b'f', b'f*', b'F', b'n'] + [b'm', b'l', b'c', b'v', b'y', b'h', b're'] + [b'sh']
    if to_delete & ObjectDeletionFlag.TEXT:
        jump_operators = [b'Tj', b'TJ', b"'", b'"']

    def clean(content: ContentStream, images: List[str], forms: List[str]) -> None:
        nonlocal jump_operators, to_delete
        i = 0
        while i < len(content.operations):
            operands, operator = content.operations[i]
            if operator == b'INLINE IMAGE' and to_delete & ObjectDeletionFlag.INLINE_IMAGES or operator in jump_operators or (operator == b'Do' and to_delete & ObjectDeletionFlag.XOBJECT_IMAGES and (operands[0] in images)):
                del content.operations[i]
            else:
                i += 1
        content.get_data()

    def clean_forms(elt: DictionaryObject, stack: List[DictionaryObject]) -> Tuple[List[str], List[str]]:
        nonlocal to_delete
        if elt in stack or (hasattr(elt, 'indirect_reference') and any((elt.indirect_reference == getattr(x, 'indirect_reference', -1) for x in stack))):
            return ([], [])
        try:
            d = cast(Dict[Any, Any], cast(DictionaryObject, elt['/Resources'])['/XObject'])
        except KeyError:
            d = {}
        images = []
        forms = []
        for k, v in d.items():
            o = v.get_object()
            try:
                content: Any = None
                if to_delete & ObjectDeletionFlag.XOBJECT_IMAGES and o['/Subtype'] == '/Image':
                    content = NullObject()
                    images.append(k)
                if o['/Subtype'] == '/Form':
                    forms.append(k)
                    if isinstance(o, ContentStream):
                        content = o
                    else:
                        content = ContentStream(o, self)
                        content.update({k1: v1 for k1, v1 in o.items() if k1 not in ['/Length', '/Filter', '/DecodeParms']})
                        try:
                            content.indirect_reference = o.indirect_reference
                        except AttributeError:
                            pass
                    stack.append(elt)
                    clean_forms(content, stack)
                if content is not None:
                    if isinstance(v, IndirectObject):
                        self._objects[v.idnum - 1] = content
                    else:
                        d[k] = self._add_object(content)
            except (TypeError, KeyError):
                pass
        for im in images:
            del d[im]
        if isinstance(elt, StreamObject):
            if not isinstance(elt, ContentStream):
                e = ContentStream(elt, self)
                e.update(elt.items())
                elt = e
            clean(elt, images, forms)
        return (images, forms)
    if not isinstance(page, PageObject):
        page = PageObject(self, page.indirect_reference)
    if '/Contents' in page:
        content = cast(ContentStream, page.get_contents())
        images, forms = clean_forms(page, [])
        clean(content, images, forms)
        page.replace_contents(content)