from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setStrokeColorCMYK(self, c, m, y, k, alpha=None):
    """set the stroke color useing negative color values
            (cyan, magenta, yellow and darkness value).
            Takes 4 arguments between 0.0 and 1.0"""
    self.setStrokeColor((c, m, y, k), alpha=alpha)