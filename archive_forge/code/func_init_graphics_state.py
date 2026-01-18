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
def init_graphics_state(self):
    self._x = 0
    self._y = 0
    self._fontname = self._initialFontName
    self._fontsize = self._initialFontSize
    self._textMode = 0
    self._leading = self._initialLeading
    self._currentMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    self._fillMode = FILL_EVEN_ODD
    self._charSpace = 0
    self._wordSpace = 0
    self._horizScale = 100
    self._textRenderMode = 0
    self._rise = 0
    self._textLineMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    self._textMatrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    self._lineCap = 0
    self._lineJoin = 0
    self._lineDash = None
    self._lineWidth = 1
    self._mitreLimit = 0
    self._fillColorObj = self._strokeColorObj = rl_config.canvas_baseColor or (0, 0, 0)
    self._extgstate = ExtGState()