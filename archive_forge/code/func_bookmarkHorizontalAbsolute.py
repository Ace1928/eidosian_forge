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
def bookmarkHorizontalAbsolute(self, key, top, left=0, fit='XYZ', **kw):
    """Bind a bookmark (destination) to the current page at a horizontal position.
           Note that the yhorizontal of the book mark is with respect to the default
           user space (where the origin is at the lower left corner of the page)
           and completely ignores any transform (translation, scale, skew, rotation,
           etcetera) in effect for the current graphics state.  The programmer is
           responsible for making sure the bookmark matches an appropriate item on
           the page."""
    return self.bookmarkPage(key, fit=fit, top=top, left=left, zoom=0)