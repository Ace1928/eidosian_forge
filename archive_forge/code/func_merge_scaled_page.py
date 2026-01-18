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
def merge_scaled_page(self, page2: 'PageObject', scale: float, over: bool=True, expand: bool=False) -> None:
    """
        merge_scaled_page is similar to merge_page, but the stream to be merged
        is scaled by applying a transformation matrix.

        Args:
          page2: The page to be merged into this one.
          scale: The scaling factor
          over: set the page2 content over page1 if True(default) else under
          expand: Whether the page should be expanded to fit the
            dimensions of the page to be merged.
        """
    op = Transformation().scale(scale, scale)
    self.merge_transformed_page(page2, op, over, expand)