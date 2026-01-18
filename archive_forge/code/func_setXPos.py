from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def setXPos(self, dx):
    """Starts a new line dx away from the start of the
        current line - NOT from the current point! So if
        you call it in mid-sentence, watch out."""
    self.moveCursor(dx, 0)