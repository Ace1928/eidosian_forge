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
def check_minmax_bool1(self, pyfunc, flags):
    cfunc = jit((types.bool_, types.bool_), **flags)(pyfunc)
    operands = (False, True)
    for x, y in itertools.product(operands, operands):
        self.assertPreciseEqual(cfunc(x, y), pyfunc(x, y))