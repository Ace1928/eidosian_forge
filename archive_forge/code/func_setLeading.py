from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setLeading(self, leading):
    """How far to move down at the end of a line."""
    self._leading = leading
    self._code.append('%s TL' % fp_str(leading))