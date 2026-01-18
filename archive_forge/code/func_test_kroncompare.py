import numpy as np
import functools
import sys
import pytest
from numpy.lib.shape_base import (
from numpy.testing import (
def test_kroncompare(self):
    from numpy.random import randint
    reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
    shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
    for s in shape:
        b = randint(0, 10, size=s)
        for r in reps:
            a = np.ones(r, b.dtype)
            large = tile(b, r)
            klarge = kron(a, b)
            assert_equal(large, klarge)