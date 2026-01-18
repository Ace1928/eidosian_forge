import warnings
import numpy as np
from numba import typed, njit, errors, typeof
from numba.core import types
from numba.experimental import structref
from numba.extending import overload_method, overload_attribute
from numba.tests.support import (
def assert_not_cached(stats):
    self.assertEqual(len(stats.cache_hits), 0)
    self.assertEqual(len(stats.cache_misses), 1)