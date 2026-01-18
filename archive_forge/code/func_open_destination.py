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
@open_destination.setter
def open_destination(self, dest: Union[None, str, Destination, PageObject]) -> None:
    if dest is None:
        try:
            del self._root_object['/OpenAction']
        except KeyError:
            pass
    elif isinstance(dest, str):
        self._root_object[NameObject('/OpenAction')] = TextStringObject(dest)
    elif isinstance(dest, Destination):
        self._root_object[NameObject('/OpenAction')] = dest.dest_array
    elif isinstance(dest, PageObject):
        self._root_object[NameObject('/OpenAction')] = Destination('Opening', dest.indirect_reference if dest.indirect_reference is not None else NullObject(), PAGE_FIT).dest_array