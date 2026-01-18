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
def test_isobj_functions(self):

    def values():
        yield 1
        yield (1 + 0j)
        yield np.asarray([3, 1 + 0j, True])
        yield 'hello world'

    @jit(nopython=True)
    def optional_fn(x, cond, cfunc):
        y = x if cond else None
        return cfunc(y)
    pyfuncs = [iscomplexobj, isrealobj]
    for pyfunc in pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)
        for x in values():
            expected = pyfunc(x)
            got = cfunc(x)
            self.assertEqual(expected, got)
            expected_optional = optional_fn.py_func(x, True, pyfunc)
            got_optional = optional_fn(x, True, cfunc)
            self.assertEqual(expected_optional, got_optional)
            expected_none = optional_fn.py_func(x, False, pyfunc)
            got_none = optional_fn(x, False, cfunc)
            self.assertEqual(expected_none, got_none)
        self.assertEqual(len(cfunc.signatures), 8)