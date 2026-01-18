from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('f_min_rtol', [0.001, 1e-05, 1e-07])
@pytest.mark.parametrize('locally_biased', [True, False])
def test_f_min(self, f_min_rtol, locally_biased):
    f_min = 1.0
    bounds = 4 * [(-2.0, 10.0)]
    res = direct(self.sphere, bounds=bounds, f_min=f_min, f_min_rtol=f_min_rtol, locally_biased=locally_biased)
    assert res.status == 3
    assert res.success
    assert res.fun < f_min * (1.0 + f_min_rtol)
    message = f'The best function value found is within a relative error={f_min_rtol} of the (known) global optimum f_min'
    assert res.message == message