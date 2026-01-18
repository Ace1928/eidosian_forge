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
def add_page(self, page: PageObject, excluded_keys: Iterable[str]=()) -> PageObject:
    """
        Add a page to this PDF file.

        Recommended for advanced usage including the adequate excluded_keys.

        The page is usually acquired from a :class:`PdfReader<pypdf.PdfReader>`
        instance.

        Args:
            page: The page to add to the document. Should be
                an instance of :class:`PageObject<pypdf._page.PageObject>`
            excluded_keys:

        Returns:
            The added PageObject.
        """
    return self._add_page(page, list.append, excluded_keys)