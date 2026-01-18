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
def linearGradient(self, x0, y0, x1, y1, colors, positions=None, extend=True):
    from reportlab.pdfbase.pdfdoc import PDFAxialShading
    colorSpace, ncolors = _normalizeColors(colors)
    fcn = _buildColorFunction(ncolors, positions)
    shading = PDFAxialShading(x0, y0, x1, y1, Function=fcn, ColorSpace=colorSpace, Extend=_gradientExtendStr(extend))
    self.shade(shading)