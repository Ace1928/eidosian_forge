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
def find_outline_item(self, outline_item: Dict[str, Any], root: Optional[OutlineType]=None) -> Optional[List[int]]:
    if root is None:
        o = self.get_outline_root()
    else:
        o = cast('TreeObject', root)
    i = 0
    while o is not None:
        if o.indirect_reference == outline_item or o.get('/Title', None) == outline_item:
            return [i]
        elif '/First' in o:
            res = self.find_outline_item(outline_item, cast(OutlineType, o['/First']))
            if res:
                return ([i] if '/Title' in o else []) + res
        if '/Next' in o:
            i += 1
            o = cast(TreeObject, o['/Next'])
        else:
            return None