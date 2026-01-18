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
def setFont(self, psfontname, size, leading=None):
    """Sets the font.  If leading not specified, defaults to 1.2 x
        font size. Raises a readable exception if an illegal font
        is supplied.  Font names are case-sensitive! Keeps track
        of font name and size for metrics."""
    self._fontname = psfontname
    self._fontsize = size
    if leading is None:
        leading = size * 1.2
    self._leading = leading
    font = pdfmetrics.getFont(self._fontname)
    if not font._dynamicFont:
        if font.face.builtIn or not getattr(self, '_drawTextAsPath', False):
            pdffontname = self._doc.getInternalFontName(psfontname)
            self._code.append('BT %s %s Tf %s TL ET' % (pdffontname, fp_str(size), fp_str(leading)))