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
def generate_file_identifiers(self) -> None:
    """
        Generate an identifier for the PDF that will be written.

        The only point of this is ensuring uniqueness. Reproducibility is not
        required.
        When a file is first written, both identifiers shall be set to the same value.
        If both identifiers match when a file reference is resolved, it is very
        likely that the correct and unchanged file has been found. If only the first
        identifier matches, a different version of the correct file has been found.
        see 14.4 "File Identifiers".
        """
    if self._ID:
        id1 = self._ID[0]
        id2 = self._compute_document_identifier()
    else:
        id1 = self._compute_document_identifier()
        id2 = id1
    self._ID = ArrayObject((id1, id2))