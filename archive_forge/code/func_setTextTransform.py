from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setTextTransform(self, a, b, c, d, e, f):
    """Like setTextOrigin, but does rotation, scaling etc."""
    if not self._canvas.bottomup:
        c = -c
        d = -d
    self._code.append('%s Tm' % fp_str(a, b, c, d, e, f))
    self._x0 = self._x = e
    self._y0 = self._y = f