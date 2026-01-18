from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setRise(self, rise):
    """Move text baseline up or down to allow superscript/subscripts"""
    self._rise = rise
    self._y = self._y - rise
    self._code.append('%s Ts' % fp_str(rise))