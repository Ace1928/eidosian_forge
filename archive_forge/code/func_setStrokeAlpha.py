from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setStrokeAlpha(self, a):
    if not (isinstance(a, (float, int)) and 0 <= a <= 1):
        raise ValueError('setStrokeAlpha invalid value %r' % a)
    getattr(self, '_setStrokeAlpha', lambda x: None)(a)