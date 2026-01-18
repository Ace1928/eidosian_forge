import math
import numpy as np
import numbers
import re
import traceback
import multiprocessing as mp
import numba
from numba import njit, prange
from numba.core import config
from numba.tests.support import TestCase, tag, override_env_config
import unittest
def func_patterns(func, args, res, dtype, mode, vlen, fastmath, pad=' ' * 8):
    """
    For a given function and its usage modes,
    returns python code and assembly patterns it should and should not generate
    """
    if mode == 'scalar':
        arg_list = ','.join([a + '[0]' for a in args])
        body = '%s%s[0] += math.%s(%s)\n' % (pad, res, func, arg_list)
    elif mode == 'numpy':
        body = '%s%s += np.%s(%s)' % (pad, res, func, ','.join(args))
        body += '.astype(np.%s)\n' % dtype if dtype.startswith('int') else '\n'
    else:
        assert mode == 'range' or mode == 'prange'
        arg_list = ','.join([a + '[i]' for a in args])
        body = '{pad}for i in {mode}({res}.size):\n{pad}{pad}{res}[i] += math.{func}({arg_list})\n'.format(**locals())
    is_f32 = dtype == 'float32' or dtype == 'complex64'
    f = func + 'f' if is_f32 else func
    v = vlen * 2 if is_f32 else vlen
    prec_suff = '' if fastmath else '_ha'
    scalar_func = '$_' + f if config.IS_OSX else '$' + f
    svml_func = '__svml_%s%d%s,' % (f, v, prec_suff)
    if mode == 'scalar':
        contains = [scalar_func]
        avoids = ['__svml_', svml_func]
    else:
        contains = [svml_func]
        avoids = []
        if vlen != 8 and (is_f32 or dtype == 'int32'):
            avoids += ['%zmm', '__svml_%s%d%s,' % (f, v * 2, prec_suff)]
    return (body, contains, avoids)