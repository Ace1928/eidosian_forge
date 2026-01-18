import operator
import functools
import warnings
import numpy as np
from numpy.core.multiarray import dragon4_positional, dragon4_scientific
from numpy.core.umath import absolute
def format_float(x, parens=False):
    if not np.issubdtype(type(x), np.floating):
        return str(x)
    opts = np.get_printoptions()
    if np.isnan(x):
        return opts['nanstr']
    elif np.isinf(x):
        return opts['infstr']
    exp_format = False
    if x != 0:
        a = absolute(x)
        if a >= 100000000.0 or a < 10 ** min(0, -(opts['precision'] - 1) // 2):
            exp_format = True
    trim, unique = ('0', True)
    if opts['floatmode'] == 'fixed':
        trim, unique = ('k', False)
    if exp_format:
        s = dragon4_scientific(x, precision=opts['precision'], unique=unique, trim=trim, sign=opts['sign'] == '+')
        if parens:
            s = '(' + s + ')'
    else:
        s = dragon4_positional(x, precision=opts['precision'], fractional=True, unique=unique, trim=trim, sign=opts['sign'] == '+')
    return s