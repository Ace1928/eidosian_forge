from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setFillColorRGB(self, r, g, b, alpha=None):
    """Set the fill color using positive color description
           (Red,Green,Blue).  Takes 3 arguments between 0.0 and 1.0"""
    self.setFillColor((r, g, b), alpha=alpha)