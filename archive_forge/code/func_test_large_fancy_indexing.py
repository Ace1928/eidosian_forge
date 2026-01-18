import os
import numpy as np
from numpy.testing import (
def test_large_fancy_indexing(self):
    nbits = np.dtype(np.intp).itemsize * 8
    thesize = int((2 ** nbits) ** (1.0 / 5.0) + 1)

    def dp():
        n = 3
        a = np.ones((n,) * 5)
        i = np.random.randint(0, n, size=thesize)
        a[np.ix_(i, i, i, i, i)] = 0

    def dp2():
        n = 3
        a = np.ones((n,) * 5)
        i = np.random.randint(0, n, size=thesize)
        a[np.ix_(i, i, i, i, i)]
    assert_raises(ValueError, dp)
    assert_raises(ValueError, dp2)