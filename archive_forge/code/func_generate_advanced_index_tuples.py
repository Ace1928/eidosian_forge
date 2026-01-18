import itertools
import numpy as np
import unittest
from numba import jit, typeof, njit
from numba.core import types
from numba.core.errors import TypingError
from numba.tests.support import MemoryLeakMixin, TestCase
def generate_advanced_index_tuples(self, N, maxdim, many=True):
    """
        Generate advanced index tuples by generating basic index tuples
        and adding a single advanced index item.
        """
    choices = list(self.generate_advanced_indices(N, many=many))
    for i in range(maxdim + 1):
        for tup in self.generate_basic_index_tuples(N, maxdim - 1, many):
            for adv in choices:
                yield (tup[:i] + (adv,) + tup[i:])