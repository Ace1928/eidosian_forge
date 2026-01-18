import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def rgb2cmyk(r, g, b):
    """one way to get cmyk from rgb"""
    c = 1 - r
    m = 1 - g
    y = 1 - b
    k = min(c, m, y)
    c = min(1, max(0, c - k))
    m = min(1, max(0, m - k))
    y = min(1, max(0, y - k))
    k = min(1, max(0, k))
    return (c, m, y, k)