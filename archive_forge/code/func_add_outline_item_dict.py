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
def add_outline_item_dict(self, outline_item: OutlineItemType, parent: Union[None, TreeObject, IndirectObject]=None, before: Union[None, TreeObject, IndirectObject]=None, is_open: bool=True) -> IndirectObject:
    outline_item_object = TreeObject()
    outline_item_object.update(outline_item)
    if '/A' in outline_item:
        action = DictionaryObject()
        a_dict = cast(DictionaryObject, outline_item['/A'])
        for k, v in list(a_dict.items()):
            action[NameObject(str(k))] = v
        action_ref = self._add_object(action)
        outline_item_object[NameObject('/A')] = action_ref
    return self.add_outline_item_destination(outline_item_object, parent, before, is_open)