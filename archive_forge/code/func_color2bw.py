import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def color2bw(colorRGB):
    """Transform an RGB color to a black and white equivalent."""
    col = colorRGB
    r, g, b, a = (col.red, col.green, col.blue, col.alpha)
    n = (r + g + b) / 3.0
    bwColorRGB = Color(n, n, n, a)
    return bwColorRGB