import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def rgbVal(self, v):
    v = v.strip()
    try:
        c = float(v)
        if 0 < c <= 1:
            c *= 255
        return int(min(255, max(0, c))) / 255.0
    except:
        raise ValueError('bad argument value %r in css color %r' % (v, self.s))