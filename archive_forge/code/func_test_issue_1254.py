import numpy as np
import unittest
from numba import jit, njit
from numba.core import types
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.core.datamodel.testing import test_factory
def test_issue_1254(self):
    """
        Missing environment for returning array
        """

    @jit(nopython=True)
    def random_directions(n):
        for i in range(n):
            vec = np.empty(3)
            vec[:] = 12
            yield vec
    outputs = list(random_directions(5))
    self.assertEqual(len(outputs), 5)
    expect = np.empty(3)
    expect[:] = 12
    for got in outputs:
        np.testing.assert_equal(expect, got)