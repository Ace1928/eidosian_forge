from statsmodels.compat import lrange
import os
import numpy as np
import pytest
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
import statsmodels.genmod.generalized_estimating_equations as gee
import statsmodels.tools as tools
import statsmodels.regression.linear_model as lm
from statsmodels.genmod import families
from statsmodels.genmod import cov_struct
import statsmodels.discrete.discrete_model as discrete
import pandas as pd
from scipy.stats.distributions import norm
import warnings
def test_grid_ar():
    np.random.seed(243)
    r = 0.5
    m = 10
    ng = 100
    ii = np.arange(m)
    cov = r ** np.abs(np.subtract.outer(ii, ii))
    covr = np.linalg.cholesky(cov)
    e = [np.dot(covr, np.random.normal(size=m)) for k in range(ng)]
    e = 2 * np.concatenate(e)
    grps = [[k] * m for k in range(ng)]
    grps = np.concatenate(grps)
    x = np.random.normal(size=(ng * m, 3))
    y = np.dot(x, np.r_[1, -1, 0]) + e
    model1 = gee.GEE(y, x, groups=grps, cov_struct=cov_struct.Autoregressive(grid=False))
    result1 = model1.fit()
    model2 = gee.GEE(y, x, groups=grps, cov_struct=cov_struct.Autoregressive(grid=True))
    result2 = model2.fit()
    model3 = gee.GEE(y, x, groups=grps, cov_struct=cov_struct.Stationary(max_lag=1, grid=False))
    result3 = model3.fit()
    assert_allclose(result1.cov_struct.dep_params, result2.cov_struct.dep_params, rtol=0.05)
    assert_allclose(result1.cov_struct.dep_params, result3.cov_struct.dep_params[1], rtol=0.05)