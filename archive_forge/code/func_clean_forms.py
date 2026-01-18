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