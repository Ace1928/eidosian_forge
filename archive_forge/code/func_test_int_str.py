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
def test_int_str(self):
    pyfunc = str_usecase
    small_inputs = [1234, 1, 0, 10, 1000]
    large_inputs = [123456789, 2222222, 1000000, ~0]
    args = [*small_inputs, *large_inputs]
    typs = [types.int8, types.int16, types.int32, types.int64, types.uint, types.uint8, types.uint16, types.uint32, types.uint64]
    for typ in typs:
        cfunc = jit((typ,), **nrt_no_pyobj_flags)(pyfunc)
        for v in args:
            self.assertPreciseEqual(cfunc(typ(v)), pyfunc(typ(v)))
            if typ.signed:
                self.assertPreciseEqual(cfunc(typ(-v)), pyfunc(typ(-v)))