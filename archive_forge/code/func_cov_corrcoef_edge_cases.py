import itertools
import math
import platform
from functools import partial
from itertools import product
import warnings
from textwrap import dedent
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.typed import List, Dict
from numba.np.numpy_support import numpy_version
from numba.core.errors import TypingError, NumbaDeprecationWarning
from numba.core.config import IS_32BITS
from numba.core.utils import pysignature
from numba.np.extensions import cross2d
from numba.tests.support import (TestCase, MemoryLeakMixin,
import unittest
def cov_corrcoef_edge_cases(self, pyfunc, first_arg_name):
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc, abs_tol=1e-14)
    m = np.array([-2.1, -1, 4.3])
    y = np.array([3, 1.1, 0.12])
    params = {first_arg_name: m, 'y': y}
    _check(params)
    m = np.array([1, 2, 3])
    y = np.array([[1j, 2j, 3j]])
    params = {first_arg_name: m, 'y': y}
    _check(params)
    m = np.array([1, 2, 3])
    y = (1j, 2j, 3j)
    params = {first_arg_name: m, 'y': y}
    _check(params)
    params = {first_arg_name: y, 'y': m}
    _check(params)
    m = np.array([1, 2, 3])
    y = (1j, 2j, 3)
    params = {first_arg_name: m, 'y': y}
    _check(params)
    params = {first_arg_name: y, 'y': m}
    _check(params)
    m = np.array([])
    y = np.array([])
    params = {first_arg_name: m, 'y': y}
    _check(params)
    m = 1.1
    y = 2.2
    params = {first_arg_name: m, 'y': y}
    _check(params)
    m = self.rnd.randn(10, 3)
    y = np.array([-2.1, -1, 4.3]).reshape(1, 3) / 10
    params = {first_arg_name: m, 'y': y}
    _check(params)
    m = np.array([-2.1, -1, 4.3])
    y = np.array([[3, 1.1, 0.12], [3, 1.1, 0.12]])
    params = {first_arg_name: m, 'y': y}
    _check(params)
    for rowvar in (False, True):
        m = np.array([-2.1, -1, 4.3])
        y = np.array([[3, 1.1, 0.12], [3, 1.1, 0.12], [4, 1.1, 0.12]])
        params = {first_arg_name: m, 'y': y, 'rowvar': rowvar}
        _check(params)
        params = {first_arg_name: y, 'y': m, 'rowvar': rowvar}
        _check(params)