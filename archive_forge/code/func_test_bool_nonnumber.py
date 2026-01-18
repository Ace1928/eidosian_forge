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
def test_bool_nonnumber(self, flags=forceobj_flags):
    pyfunc = bool_usecase
    cfunc = jit((types.string,), **flags)(pyfunc)
    for x in ['x', '']:
        self.assertPreciseEqual(cfunc(x), pyfunc(x))
    cfunc = jit((types.Dummy('list'),), **flags)(pyfunc)
    for x in [[1], []]:
        self.assertPreciseEqual(cfunc(x), pyfunc(x))