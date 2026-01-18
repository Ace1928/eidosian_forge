from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setTextOrigin(self, x, y):
    if self._canvas.bottomup:
        self._code.append('1 0 0 1 %s Tm' % fp_str(x, y))
    else:
        self._code.append('1 0 0 -1 %s Tm' % fp_str(x, y))
    self._x0 = self._x = x
    self._y0 = self._y = y