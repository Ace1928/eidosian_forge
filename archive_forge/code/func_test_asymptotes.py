import numpy as np
from numpy.testing import assert_equal, assert_allclose
import scipy.special as sc
def test_asymptotes(self):
    assert_equal(sc.ndtri([0.0, 1.0]), [-np.inf, np.inf])