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
@property
def page_number(self) -> Optional[int]:
    """
        Read-only property which return the page number with the pdf file.

        Returns:
            int : page number ; None if the page is not attached to a pdf
        """
    if self.indirect_reference is None:
        return None
    else:
        try:
            lst = self.indirect_reference.pdf.pages
            return lst.index(self)
        except ValueError:
            return None