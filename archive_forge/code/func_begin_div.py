import io
import logging
import re
from typing import (
from pdfminer.pdfcolor import PDFColorSpace
from . import utils
from .image import ImageWriter
from .layout import LAParams, LTComponent, TextGroupElement
from .layout import LTAnno
from .layout import LTChar
from .layout import LTContainer
from .layout import LTCurve
from .layout import LTFigure
from .layout import LTImage
from .layout import LTItem
from .layout import LTLayoutContainer
from .layout import LTLine
from .layout import LTPage
from .layout import LTRect
from .layout import LTText
from .layout import LTTextBox
from .layout import LTTextBoxVertical
from .layout import LTTextGroup
from .layout import LTTextLine
from .pdfdevice import PDFTextDevice
from .pdffont import PDFFont
from .pdffont import PDFUnicodeNotDefined
from .pdfinterp import PDFGraphicState, PDFResourceManager
from .pdfpage import PDFPage
from .pdftypes import PDFStream
from .utils import AnyIO, Point, Matrix, Rect, PathSegment, make_compat_str
from .utils import apply_matrix_pt
from .utils import bbox2str
from .utils import enc
from .utils import mult_matrix
def begin_div(self, color: str, borderwidth: int, x: float, y: float, w: float, h: float, writing_mode: str='False') -> None:
    self._fontstack.append(self._font)
    self._font = None
    s = '<div style="position:absolute; border: %s %dpx solid; writing-mode:%s; left:%dpx; top:%dpx; width:%dpx; height:%dpx;">' % (color, borderwidth, writing_mode, x * self.scale, (self._yoffset - y) * self.scale, w * self.scale, h * self.scale)
    self.write(s)
    return