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
def clone_reader_document_root(self, reader: PdfReader) -> None:
    """
        Copy the reader document root to the writer and all sub elements,
        including pages, threads, outlines,... For partial insertion, ``append``
        should be considered.

        Args:
            reader: PdfReader from the document root should be copied.
        """
    self._objects.clear()
    self._root_object = reader.root_object.clone(self)
    self._root = self._root_object.indirect_reference
    self._pages = self._root_object.raw_get('/Pages')
    self._flatten()
    assert self.flattened_pages is not None
    for p in self.flattened_pages:
        p[NameObject('/Parent')] = self._pages
        self._objects[cast(IndirectObject, p.indirect_reference).idnum - 1] = p
    cast(DictionaryObject, self._pages.get_object())[NameObject('/Kids')] = ArrayObject([p.indirect_reference for p in self.flattened_pages])