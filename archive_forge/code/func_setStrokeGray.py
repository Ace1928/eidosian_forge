from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setStrokeGray(self, gray, alpha=None):
    """Sets the gray level; 0.0=black, 1.0=white"""
    self._strokeColorObj = (gray, gray, gray)
    self._code.append('%s G' % fp_str(gray))
    if alpha is not None:
        self.setFillAlpha(alpha)