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
def clean_page(self, page: Union[PageObject, IndirectObject]) -> PageObject:
    """
        Perform some clean up in the page.
        Currently: convert NameObject nameddestination to TextStringObject
        (required for names/dests list)

        Args:
            page:

        Returns:
            The cleaned PageObject
        """
    page = cast('PageObject', page.get_object())
    for a in page.get('/Annots', []):
        a_obj = a.get_object()
        d = a_obj.get('/Dest', None)
        act = a_obj.get('/A', None)
        if isinstance(d, NameObject):
            a_obj[NameObject('/Dest')] = TextStringObject(d)
        elif act is not None:
            act = act.get_object()
            d = act.get('/D', None)
            if isinstance(d, NameObject):
                act[NameObject('/D')] = TextStringObject(d)
    return page