import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.regression.linear_model import OLS
import statsmodels.stats.power as smpwr
import statsmodels.stats.oneway as smo  # needed for function with `test`
from statsmodels.stats.oneway import (
from statsmodels.stats.robust_compare import scale_transform
from statsmodels.stats.contrast import (
def test_effectsize_power():
    n_groups = 3
    means = [527.86, 660.43, 649.14]
    vars_ = 107.4304 ** 2
    nobs = 12
    es = effectsize_oneway(means, vars_, nobs, use_var='equal', ddof_between=0)
    es = np.sqrt(es)
    alpha = 0.05
    power = 0.8
    nobs_t = nobs * n_groups
    kwds = {'effect_size': es, 'nobs': nobs_t, 'alpha': alpha, 'power': power, 'k_groups': n_groups}
    from statsmodels.stats.power import FTestAnovaPower
    res_pow = 0.8251
    res_es = 0.559
    kwds_ = kwds.copy()
    del kwds_['power']
    p = FTestAnovaPower().power(**kwds_)
    assert_allclose(p, res_pow, atol=0.0001)
    assert_allclose(es, res_es, atol=0.0006)
    nobs = np.array([15, 9, 9])
    kwds['nobs'] = nobs
    es = effectsize_oneway(means, vars_, nobs, use_var='equal', ddof_between=0)
    es = np.sqrt(es)
    kwds['effect_size'] = es
    p = FTestAnovaPower().power(**kwds_)
    res_pow = 0.8297
    res_es = 0.59
    assert_allclose(p, res_pow, atol=0.005)
    assert_allclose(es, res_es, atol=0.0006)