import numpy as np
from numpy.testing import (assert_equal,
import pytest
import scipy.signal._wavelets as wavelets
def test_cascade(self):
    with pytest.deprecated_call():
        for J in range(1, 7):
            for i in range(1, 5):
                lpcoef = wavelets.daub(i)
                k = len(lpcoef)
                x, phi, psi = wavelets.cascade(lpcoef, J)
                assert_(len(x) == len(phi) == len(psi))
                assert_equal(len(x), (k - 1) * 2 ** J)