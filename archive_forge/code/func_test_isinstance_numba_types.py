import itertools
import functools
import sys
import operator
from collections import namedtuple
import numpy as np
import unittest
import warnings
from numba import jit, typeof, njit, typed
from numba.core import errors, types, config
from numba.tests.support import (TestCase, tag, ignore_internal_warnings,
from numba.core.extending import overload_method, box
def test_isinstance_numba_types(self):
    pyfunc = isinstance_usecase_numba_types
    cfunc = jit(nopython=True)(pyfunc)
    inputs = ((types.int32(1), 'int32'), (types.int64(2), 'int64'), (types.float32(3.0), 'float32'), (types.float64(4.0), 'float64'), (types.complex64(5j), 'no match'), (typed.List([1, 2]), 'typed list'), (typed.Dict.empty(types.int64, types.int64), 'typed dict'))
    for inpt, expected in inputs:
        got = cfunc(inpt)
        self.assertEqual(expected, got)