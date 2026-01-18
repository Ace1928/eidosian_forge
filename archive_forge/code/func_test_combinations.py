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
def test_combinations(self):

    def gen_w_arg(clazz_type):

        def impl(x):
            return isinstance(x, clazz_type)
        return impl
    clazz_types = (int, float, complex, str, list, tuple, bytes, set, range, np.int8, np.float32)
    instances = (1, 2.3, 4j, '5', [6], (7,), b'8', {9}, None, (10, 11, 12), (13, 'a', 14j), np.array([15, 16, 17]), np.int8(18), np.float32(19), typed.Dict.empty(types.unicode_type, types.float64), typed.List.empty_list(types.complex128), np.ones(4))
    for ct in clazz_types:
        fn = njit(gen_w_arg(ct))
        for x in instances:
            expected = fn.py_func(x)
            got = fn(x)
            self.assertEqual(got, expected)