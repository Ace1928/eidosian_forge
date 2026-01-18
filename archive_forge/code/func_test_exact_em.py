import numpy as np
from statsmodels.multivariate.factor import Factor
from numpy.testing import assert_allclose, assert_equal
from scipy.optimize import approx_fprime
import warnings
def test_exact_em():
    np.random.seed(23324)
    for k_var in (5, 10, 25):
        for n_factor in (1, 2, 3):
            load = np.random.normal(size=(k_var, n_factor))
            uniq = np.linspace(1, 2, k_var)
            c = np.dot(load, load.T)
            c.flat[::c.shape[0] + 1] += uniq
            s = np.sqrt(np.diag(c))
            c /= np.outer(s, s)
            fa = Factor(corr=c, n_factor=n_factor, method='ml')
            load_e, uniq_e = fa._fit_ml_em(2000)
            c_e = np.dot(load_e, load_e.T)
            c_e.flat[::c_e.shape[0] + 1] += uniq_e
            assert_allclose(c_e, c, rtol=0.0001, atol=0.0001)