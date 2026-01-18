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
def test_ediff1d_basic(self):
    pyfunc = ediff1d
    cfunc = jit(nopython=True)(pyfunc)
    _check = partial(self._check_output, pyfunc, cfunc)

    def to_variations(a):
        yield None
        yield a
        yield a.astype(np.int16)

    def ary_variations(a):
        yield a
        yield a.reshape(3, 2, 2)
        yield a.astype(np.int32)
    for ary in ary_variations(np.linspace(-2, 7, 12)):
        params = {'ary': ary}
        _check(params)
        for a in to_variations(ary):
            params = {'ary': ary, 'to_begin': a}
            _check(params)
            params = {'ary': ary, 'to_end': a}
            _check(params)
            for b in to_variations(ary):
                params = {'ary': ary, 'to_begin': a, 'to_end': b}
                _check(params)