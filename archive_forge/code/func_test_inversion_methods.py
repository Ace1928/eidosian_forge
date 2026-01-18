import numpy as np
from statsmodels.tsa.statespace.kalman_filter import (
from statsmodels.tsa.statespace.kalman_smoother import (
from statsmodels.tsa.statespace.simulation_smoother import (
from numpy.testing import assert_equal
def test_inversion_methods(self):
    model = self.model
    model.inversion_method = 0
    model.invert_univariate = True
    assert_equal(model.inversion_method, INVERT_UNIVARIATE)
    model.invert_cholesky = True
    assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)
    model.invert_univariate = False
    assert_equal(model.inversion_method, INVERT_CHOLESKY)
    model.set_inversion_method(INVERT_LU)
    assert_equal(model.inversion_method, INVERT_LU)
    model.set_inversion_method(invert_cholesky=True, invert_univariate=True, invert_lu=False)
    assert_equal(model.inversion_method, INVERT_UNIVARIATE | INVERT_CHOLESKY)
    model.inversion_method = 0
    for name in model.inversion_methods:
        setattr(model, name, True)
    assert_equal(model.inversion_method, INVERT_UNIVARIATE | SOLVE_LU | INVERT_LU | SOLVE_CHOLESKY | INVERT_CHOLESKY)
    for name in model.inversion_methods:
        setattr(model, name, False)
    assert_equal(model.inversion_method, 0)