import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
from statsmodels.gam.smooth_basis import (UnivariatePolynomialSmoother,
def test_multivariate_polynomial_basis():
    np.random.seed(1)
    x = np.random.normal(0, 1, (10, 2))
    degrees = [3, 4]
    mps = PolynomialSmoother(x, degrees)
    for i, deg in enumerate(degrees):
        uv_basis = UnivariatePolynomialSmoother(x[:, i], degree=deg).basis
        assert_allclose(mps.smoothers[i].basis, uv_basis)