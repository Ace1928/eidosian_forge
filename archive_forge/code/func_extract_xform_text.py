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
def extract_xform_text(self, xform: EncodedStreamObject, orientations: Tuple[int, ...]=(0, 90, 270, 360), space_width: float=200.0, visitor_operand_before: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_operand_after: Optional[Callable[[Any, Any, Any, Any], None]]=None, visitor_text: Optional[Callable[[Any, Any, Any, Any, Any], None]]=None) -> str:
    """
        Extract text from an XObject.

        Args:
            xform:
            orientations:
            space_width:  force default space width (if not extracted from font (default 200)
            visitor_operand_before:
            visitor_operand_after:
            visitor_text:

        Returns:
            The extracted text
        """
    return self._extract_text(xform, self.pdf, orientations, space_width, None, visitor_operand_before, visitor_operand_after, visitor_text)