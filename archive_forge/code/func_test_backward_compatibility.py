import numpy as np
from numpy.testing import assert_allclose, assert_equal
import statsmodels.base._penalties as smpen
from statsmodels.tools.numdiff import approx_fprime, approx_hess
def test_backward_compatibility(self):
    wts = [0.5]
    pen = smpen.L2(weights=wts)
    assert_equal(pen.weights, wts)