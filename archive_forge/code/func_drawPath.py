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
def drawPath(self, aPath, stroke=1, fill=0, fillMode=None):
    """Draw the path object in the mode indicated"""
    if fillMode is None:
        fillMode = getattr(aPath, '_fillMode', self._fillMode)
    self._code.append(str(aPath.getCode()))
    self._strokeAndFill(stroke, fill, fillMode)