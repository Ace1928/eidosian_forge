import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def hsl2rgb(h, s, l):
    if l <= 0.5:
        m2 = l * (s + 1)
    else:
        m2 = l + s - l * s
    m1 = l * 2 - m2
    return (hue2rgb(m1, m2, h + 1.0 / 3), hue2rgb(m1, m2, h), hue2rgb(m1, m2, h - 1.0 / 3))