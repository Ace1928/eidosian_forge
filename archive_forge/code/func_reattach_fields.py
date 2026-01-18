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
def reattach_fields(self, page: Optional[PageObject]=None) -> List[DictionaryObject]:
    """
        Parse annotations within the page looking for orphan fields and
        reattach then into the Fields Structure.

        Args:
            page: page to analyze.
                  If none is provided, all pages will be analyzed.

        Returns:
            list of reattached fields.
        """
    lst = []
    if page is None:
        for p in self.pages:
            lst += self.reattach_fields(p)
        return lst
    try:
        af = cast(DictionaryObject, self._root_object[CatalogDictionary.ACRO_FORM])
    except KeyError:
        af = DictionaryObject()
        self._root_object[NameObject(CatalogDictionary.ACRO_FORM)] = af
    try:
        fields = cast(ArrayObject, af[InteractiveFormDictEntries.Fields])
    except KeyError:
        fields = ArrayObject()
        af[NameObject(InteractiveFormDictEntries.Fields)] = fields
    if '/Annots' not in page:
        return lst
    annots = cast(ArrayObject, page['/Annots'])
    for idx in range(len(annots)):
        ano = annots[idx]
        indirect = isinstance(ano, IndirectObject)
        ano = cast(DictionaryObject, ano.get_object())
        if ano.get('/Subtype', '') == '/Widget' and '/FT' in ano:
            if 'indirect_reference' in ano.__dict__ and ano.indirect_reference in fields:
                continue
            if not indirect:
                annots[idx] = self._add_object(ano)
            fields.append(ano.indirect_reference)
            lst.append(ano)
    return lst