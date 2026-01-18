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
def bezier(self, x1, y1, x2, y2, x3, y3, x4, y4):
    """Bezier curve with the four given control points"""
    self._code.append('n %s m %s c S' % (fp_str(x1, y1), fp_str(x2, y2, x3, y3, x4, y4)))