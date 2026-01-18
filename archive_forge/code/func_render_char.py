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
def render_char(self, matrix: Matrix, font: PDFFont, fontsize: float, scaling: float, rise: float, cid: int, ncs: PDFColorSpace, graphicstate: PDFGraphicState) -> float:
    try:
        text = font.to_unichr(cid)
        assert isinstance(text, str), str(type(text))
    except PDFUnicodeNotDefined:
        text = self.handle_undefined_char(font, cid)
    textwidth = font.char_width(cid)
    textdisp = font.char_disp(cid)
    item = LTChar(matrix, font, fontsize, scaling, rise, text, textwidth, textdisp, ncs, graphicstate)
    self.cur_item.add(item)
    return item.adv