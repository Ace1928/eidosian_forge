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
def check_minmax_1(self, pyfunc, flags):
    cfunc = jit((types.int32, types.int32), **flags)(pyfunc)
    x_operands = [-1, 0, 1]
    y_operands = [-1, 0, 1]
    for x, y in itertools.product(x_operands, y_operands):
        self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))