import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def cKwds(self):
    K = self._cKwds
    S = K[:6]
    for k in self._cKwds:
        v = getattr(self, k)
        if k in S:
            v *= 100
        yield (k, v)