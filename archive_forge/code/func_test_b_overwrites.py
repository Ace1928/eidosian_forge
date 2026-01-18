import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_b_overwrites(self):
    f = getattr(fblas, 'dtrmm', None)
    if f is not None:
        for overwr in [True, False]:
            bcopy = self.b.copy()
            result = f(1.0, self.a, bcopy, overwrite_b=overwr)
            assert_(bcopy.flags.f_contiguous is False and np.may_share_memory(bcopy, result) is False)
            assert_equal(bcopy, self.b)
        bcopy = np.asfortranarray(self.b.copy())
        result = f(1.0, self.a, bcopy, overwrite_b=True)
        assert_(bcopy.flags.f_contiguous is True and np.may_share_memory(bcopy, result) is True)
        assert_array_almost_equal(bcopy, result)