import sys
import io
from .z3consts import *
from .z3core import *
from ctypes import *
def seq2(header, args, i=4, lp='(', rp=')'):
    if len(args) == 0:
        return compose(to_format(header), to_format(lp), to_format(rp))
    else:
        return group(compose(indent(len(lp), compose(to_format(lp), to_format(header))), indent(i, compose(seq(args), to_format(rp)))))