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
def add_named_destination(self, title: str, page_number: int) -> IndirectObject:
    page_ref = self.get_object(self._pages)[PA.KIDS][page_number]
    dest = DictionaryObject()
    dest.update({NameObject(GoToActionArguments.D): ArrayObject([page_ref, NameObject(TypFitArguments.FIT_H), NumberObject(826)]), NameObject(GoToActionArguments.S): NameObject('/GoTo')})
    dest_ref = self._add_object(dest)
    if not isinstance(title, TextStringObject):
        title = TextStringObject(str(title))
    self.add_named_destination_array(title, dest_ref)
    return dest_ref