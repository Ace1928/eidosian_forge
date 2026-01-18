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
def compress_content_streams(self, level: int=-1) -> None:
    """
        Compress the size of this page by joining all content streams and
        applying a FlateDecode filter.

        However, it is possible that this function will perform no action if
        content stream compression becomes "automatic".
        """
    content = self.get_contents()
    if content is not None:
        content_obj = content.flate_encode(level)
        try:
            content.indirect_reference.pdf._objects[content.indirect_reference.idnum - 1] = content_obj
        except AttributeError:
            if self.indirect_reference is not None and hasattr(self.indirect_reference.pdf, '_add_object'):
                self.replace_contents(content_obj)
            else:
                raise ValueError('Page must be part of a PdfWriter')