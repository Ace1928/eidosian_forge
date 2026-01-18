import math, re, functools
from reportlab.lib.rl_accel import fp_str
from reportlab.lib.utils import asNative, isStr, rl_safe_eval, rl_extended_literal_eval
from reportlab import rl_config
from ast import literal_eval
import re
def getAllNamedColors():
    global _namedColors
    if _namedColors is not None:
        return _namedColors
    from reportlab.lib import colors
    _namedColors = {}
    for name, value in colors.__dict__.items():
        if isinstance(value, Color):
            _namedColors[name] = value
    return _namedColors