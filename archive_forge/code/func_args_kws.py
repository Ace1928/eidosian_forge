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
def args_kws():
    yield ([*range(10)], 12)
    yield ([x + x / 10.0 for x in range(10)], 19j)
    yield ([x * 1j for x in range(10)], -2)
    yield ((1, 2, 3), 9)
    yield ((1, 2, 3j), -0)
    yield ((np.int64(32), np.int32(2), np.int8(3)), np.uint32(7))
    tl = typed.List(range(5))
    yield (tl, 100)
    yield (np.ones((5, 5)), 10 * np.ones((5,)))
    yield (ntpl(100, 200), -50)
    yield (ntpl(100, 200j), 9)