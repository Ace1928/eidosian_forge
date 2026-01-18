import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
import scipy.special as sc
def test_compliment(self):
    x = np.linspace(-1, 1, 101)
    assert_allclose(sc.erfcinv(1 - x), sc.erfinv(x), rtol=0, atol=1e-15)