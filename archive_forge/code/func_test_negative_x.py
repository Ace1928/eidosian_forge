import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import scipy.special as sc
def test_negative_x(self):
    a, b, x = np.meshgrid([-1, -0.5, 0, 0.5, 1], [-1, -0.5, 0, 0.5, 1], np.linspace(-100, -1, 10))
    assert np.all(np.isnan(sc.hyperu(a, b, x)))