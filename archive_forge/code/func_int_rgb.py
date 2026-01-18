import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def int_rgb(self):
    v = self.bitmap_rgb()
    return v[0] << 16 | v[1] << 8 | v[2]