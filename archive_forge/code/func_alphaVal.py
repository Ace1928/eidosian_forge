import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def alphaVal(self, v, c=1, n='alpha'):
    try:
        a = float(v)
        return min(c, max(0, a))
    except:
        raise ValueError('bad %s argument value %r in css color %r' % (n, v, self.s))