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
def append_pages_from_reader(self, reader: PdfReader, after_page_append: Optional[Callable[[PageObject], None]]=None) -> None:
    """
        Copy pages from reader to writer. Includes an optional callback
        parameter which is invoked after pages are appended to the writer.

        ``append`` should be preferred.

        Args:
            reader: a PdfReader object from which to copy page
                annotations to this writer object.  The writer's annots
                will then be updated.
            after_page_append:
                Callback function that is invoked after each page is appended to
                the writer. Signature includes a reference to the appended page
                (delegates to append_pages_from_reader). The single parameter of
                the callback is a reference to the page just appended to the
                document.
        """
    reader_num_pages = len(reader.pages)
    for reader_page_number in range(reader_num_pages):
        reader_page = reader.pages[reader_page_number]
        writer_page = self.add_page(reader_page)
        if callable(after_page_append):
            after_page_append(writer_page)