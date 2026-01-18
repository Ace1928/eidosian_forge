import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
@pytest.mark.slow
def test_efficient_notrandom(self):
    nobs = 400
    np.random.seed(12345)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    Y = 0.3 + 1.2 * C1 - 0.9 * C2
    dens_efficient = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml', defaults=nparam.EstimatorSettings(efficient=True, randomize=False, n_sub=100))
    dens = nparam.KDEMultivariate(data=[Y, C1], var_type='cc', bw='cv_ml')
    npt.assert_allclose(dens.bw, dens_efficient.bw, atol=0.1, rtol=0.2)