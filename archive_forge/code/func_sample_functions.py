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
def sample_functions(op):
    yield (lambda: op(range(10)))
    yield (lambda: op(range(4, 12)))
    yield (lambda: op(range(-4, -15, -1)))
    yield (lambda: op([6.6, 5.5, 7.7]))
    yield (lambda: op([(3, 4), (1, 2)]))
    yield (lambda: op(frange(1.1, 3.3, 0.1)))
    yield (lambda: op([np.nan, -np.inf, np.inf, np.nan]))
    yield (lambda: op([(3,), (1,), (2,)]))