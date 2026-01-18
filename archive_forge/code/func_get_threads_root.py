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
def get_threads_root(self) -> ArrayObject:
    """
        The list of threads.

        See §12.4.3 of the PDF 1.7 or PDF 2.0 specification.

        Returns:
            An array (possibly empty) of Dictionaries with ``/F`` and
            ``/I`` properties.
        """
    if CO.THREADS in self._root_object:
        threads = cast(ArrayObject, self._root_object[CO.THREADS])
    else:
        threads = ArrayObject()
        self._root_object[NameObject(CO.THREADS)] = threads
    return threads