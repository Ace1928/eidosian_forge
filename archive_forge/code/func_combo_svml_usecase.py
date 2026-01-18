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
def combo_svml_usecase(dtype, mode, vlen, fastmath, name):
    """ Combine multiple function calls under single umbrella usecase """
    name = usecase_name(dtype, mode, vlen, name)
    body = 'def {name}(n):\n        x   = np.empty(n*8, dtype=np.{dtype})\n        ret = np.empty_like(x)\n'.format(**locals())
    funcs = set(numpy_funcs if mode == 'numpy' else other_funcs)
    if dtype.startswith('complex'):
        funcs = funcs.difference(complex_funcs_exclude)
    contains = set()
    avoids = set()
    for f in funcs:
        b, c, a = func_patterns(f, ['x'], 'ret', dtype, mode, vlen, fastmath)
        avoids.update(a)
        body += b
        contains.update(c)
    body += ' ' * 8 + 'return ret'
    ldict = {}
    exec(body, globals(), ldict)
    ldict[name].__doc__ = body
    return (ldict[name], contains, avoids)