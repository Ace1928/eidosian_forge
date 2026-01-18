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
def apply_on(self, pt: Union[Tuple[float, float], List[float]], as_object: bool=False) -> Union[Tuple[float, float], List[float]]:
    """
        Apply the transformation matrix on the given point.

        Args:
            pt: A tuple or list representing the point in the form (x, y)

        Returns:
            A tuple or list representing the transformed point in the form (x', y')
        """
    typ = FloatObject if as_object else float
    pt1 = (typ(float(pt[0]) * self.ctm[0] + float(pt[1]) * self.ctm[2] + self.ctm[4]), typ(float(pt[0]) * self.ctm[1] + float(pt[1]) * self.ctm[3] + self.ctm[5]))
    return list(pt1) if isinstance(pt, list) else pt1