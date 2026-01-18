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
def add_js(self, javascript: str) -> None:
    """
        Add JavaScript which will launch upon opening this PDF.

        Args:
            javascript: Your Javascript.

        >>> output.add_js("this.print({bUI:true,bSilent:false,bShrinkToFit:true});")
        # Example: This will launch the print window when the PDF is opened.
        """
    if '/Names' not in self._root_object:
        self._root_object[NameObject(CA.NAMES)] = DictionaryObject()
    names = cast(DictionaryObject, self._root_object[CA.NAMES])
    if '/JavaScript' not in names:
        names[NameObject('/JavaScript')] = DictionaryObject({NameObject('/Names'): ArrayObject()})
    js_list = cast(ArrayObject, cast(DictionaryObject, names['/JavaScript'])['/Names'])
    js = DictionaryObject()
    js.update({NameObject(PA.TYPE): NameObject('/Action'), NameObject('/S'): NameObject('/JavaScript'), NameObject('/JS'): TextStringObject(f'{javascript}')})
    js_list.append(create_string_object(str(uuid.uuid4())))
    js_list.append(self._add_object(js))