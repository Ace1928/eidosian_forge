import numpy as np
import numpy.testing as npt
import pytest
from numpy.testing import assert_allclose, assert_equal
import statsmodels.api as sm
def test_efficient_user_specified_bw(self):
    nobs = 400
    np.random.seed(12345)
    C1 = np.random.normal(size=(nobs,))
    C2 = np.random.normal(2, 1, size=(nobs,))
    bw_user = [0.23, 434697.22]
    dens = nparam.KDEMultivariate(data=[C1, C2], var_type='cc', bw=bw_user, defaults=nparam.EstimatorSettings(efficient=True, randomize=False, n_sub=100))
    npt.assert_equal(dens.bw, bw_user)