import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def cmyka(self):
    """Returns a tuple of five color components - syntactic sugar"""
    return (self.cyan, self.magenta, self.yellow, self.black, self.alpha)