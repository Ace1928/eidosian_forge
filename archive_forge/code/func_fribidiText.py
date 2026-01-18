from reportlab.lib.colors import Color, CMYKColor, CMYKColorSep, toColor
from reportlab.lib.utils import isBytes, isStr, asUnicode
from reportlab.lib.rl_accel import fp_str
from reportlab.pdfbase import pdfmetrics
from reportlab.rl_config import rtlSupport
def fribidiText(text, direction):
    return log2vis(text, directionsMap.get(direction, DIR_ON), clean=True) if direction in ('LTR', 'RTL') else text