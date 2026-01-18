import re
import hashlib
from string import digits
from math import sin, cos, tan, pi
from reportlab import rl_config
from reportlab.pdfbase import pdfdoc
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfgen  import pathobject
from reportlab.pdfgen.textobject import PDFTextObject, _PDFColorSetter
from reportlab.lib.colors import black, _chooseEnforceColorSpace, Color, CMYKColor, toColor
from reportlab.lib.utils import ImageReader, isSeq, isStr, isUnicode, _digester, asUnicode
from reportlab.lib.rl_accel import fp_str, escapePDF
from reportlab.lib.boxstuff import aspectRatioFix
def drawBoundary(self, sb, x1, y1, width, height):
    """draw a boundary as a rectangle (primarily for debugging)."""
    ss = isinstance(sb, (str, tuple, list)) or isinstance(sb, Color)
    w = -1
    da = None
    if ss:
        c = toColor(sb, -1)
        ss = c != -1
    elif isinstance(sb, ShowBoundaryValue) and sb:
        c = toColor(sb.color, -1)
        ss = c != -1
        if ss:
            w = sb.width
            da = sb.dashArray
    if ss:
        self.saveState()
        self.setStrokeColor(c)
        if w >= 0:
            self.setLineWidth(w)
        if da:
            self.setDash(da)
    self.rect(x1, y1, width, height)
    if ss:
        self.restoreState()