import gc
from io import StringIO
import numpy as np
from numba import njit, vectorize
from numba import typeof
from numba.core import utils, types, typing, ir, compiler, cpu, cgutils
from numba.core.compiler import Compiler, Flags
from numba.core.registry import cpu_target
from numba.tests.support import (MemoryLeakMixin, TestCase, temp_directory,
from numba.extending import (
import operator
import textwrap
import unittest
def test_issue_1264(self):
    n = 100
    x = np.random.uniform(size=n * 3).reshape((n, 3))
    expected = distance_matrix(x)
    actual = njit(distance_matrix)(x)
    np.testing.assert_array_almost_equal(expected, actual)
    gc.collect()