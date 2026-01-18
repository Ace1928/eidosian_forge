import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def fader(self, n, reverse=False):
    """return n colors based on density fade
        *NB* note this dosen't reach density zero"""
    scale = self._scale
    dd = scale / float(n)
    L = [self.clone(density=scale - i * dd) for i in range(n)]
    if reverse:
        L.reverse()
    return L