import numpy as _np
from .blas import _get_funcs, _memoize_get_funcs
from scipy.linalg import _flapack
from re import compile as regex_compile
from scipy.linalg._flapack import *  # noqa: E402, F403
def backtickrepl(m):
    if m.group('s'):
        return 'with bounds ``{}`` with ``{}`` storage\n'.format(m.group('b'), m.group('s'))
    else:
        return 'with bounds ``{}``\n'.format(m.group('b'))