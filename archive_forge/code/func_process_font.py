import math
import re
import sys
from decimal import Decimal
from pathlib import Path
from typing import (
from ._cmap import build_char_map, unknown_char_map
from ._protocols import PdfCommonDocProtocol
from ._text_extraction import (
from ._utils import (
from .constants import AnnotationDictionaryAttributes as ADA
from .constants import ImageAttributes as IA
from .constants import PageAttributes as PG
from .constants import Resources as RES
from .errors import PageSizeNotDefinedError, PdfReadError
from .filters import _xobj_to_image
from .generic import (
def process_font(f: DictionaryObject) -> None:
    nonlocal fnt, emb
    f = cast(DictionaryObject, f.get_object())
    if '/BaseFont' in f:
        fnt.add(cast(str, f['/BaseFont']))
    if '/CharProcs' in f or ('/FontDescriptor' in f and any((x in cast(DictionaryObject, f['/FontDescriptor']) for x in fontkeys))) or ('/DescendantFonts' in f and '/FontDescriptor' in cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object()) and any((x in cast(DictionaryObject, cast(DictionaryObject, cast(ArrayObject, f['/DescendantFonts'])[0].get_object())['/FontDescriptor']) for x in fontkeys))):
        try:
            emb.add(cast(str, f['/BaseFont']))
        except KeyError:
            emb.add('(' + cast(str, f['/Subtype']) + ')')