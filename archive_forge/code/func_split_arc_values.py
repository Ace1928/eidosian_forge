import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def split_arc_values(op, value):
    float_re = '(-?\\d*\\.?\\d*(?:[eE][+-]?\\d+)?)'
    flag_re = '([1|0])'
    a_seq_re = '[\\s,]*'.join([float_re, float_re, float_re, flag_re, flag_re, float_re, float_re]) + '[\\s,]*'
    res = []
    for seq in re.finditer(a_seq_re, value.strip()):
        res.extend([op, [float(num) for num in seq.groups()]])
    return res