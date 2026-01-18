import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings
def test_fit_ml_em_random_state():
    T = 10
    epsilon = np.random.multivariate_normal(np.zeros(3), np.eye(3), size=T).T
    initial = np.random.get_state()
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='Fitting did not converge')
        Factor(endog=epsilon, n_factor=2, method='ml').fit()
    final = np.random.get_state()
    assert initial[0] == final[0]
    assert_equal(initial[1], final[1])
    assert initial[2:] == final[2:]