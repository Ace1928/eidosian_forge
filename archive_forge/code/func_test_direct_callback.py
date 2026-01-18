from numpy.testing import (assert_allclose,
import pytest
import numpy as np
from scipy.optimize import direct, Bounds
@pytest.mark.parametrize('locally_biased', [True, False])
def test_direct_callback(self, locally_biased):
    res = direct(self.sphere, self.bounds_sphere, locally_biased=locally_biased)

    def callback(x):
        x = 2 * x
        dummy = np.square(x)
        print('DIRECT minimization algorithm callback test')
        return dummy
    res_callback = direct(self.sphere, self.bounds_sphere, locally_biased=locally_biased, callback=callback)
    assert_allclose(res.x, res_callback.x)
    assert res.nit == res_callback.nit
    assert res.nfev == res_callback.nfev
    assert res.status == res_callback.status
    assert res.success == res_callback.success
    assert res.fun == res_callback.fun
    assert_allclose(res.x, res_callback.x)
    assert res.message == res_callback.message
    assert_allclose(res_callback.x, self.optimum_sphere_pos, rtol=0.001, atol=0.001)
    assert_allclose(res_callback.fun, self.optimum_sphere, atol=1e-05, rtol=1e-05)