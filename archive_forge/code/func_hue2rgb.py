import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def hue2rgb(m1, m2, h):
    if h < 0:
        h += 1
    if h > 1:
        h -= 1
    if h * 6 < 1:
        return m1 + (m2 - m1) * h * 6
    if h * 2 < 1:
        return m2
    if h * 3 < 2:
        return m1 + (m2 - m1) * (4 - 6 * h)
    return m1