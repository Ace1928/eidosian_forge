import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def generate_basic_index_tuples(self, N, maxdim, many=True):
    """
        Generate basic index tuples with 0 to *maxdim* items.
        """
    if many:
        choices = [slice(None, None, None), slice(1, N - 1, None), slice(0, None, 2), slice(N - 1, None, -2), slice(-N + 1, -1, None), slice(-1, -N, -2)]
    else:
        choices = [slice(0, N - 1, None), slice(-1, -N, -2)]
    for ndim in range(maxdim + 1):
        for tup in itertools.product(choices, repeat=ndim):
            yield tup