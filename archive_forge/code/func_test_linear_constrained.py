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
def test_linear_constrained(self):
    family = families.Gaussian()
    np.random.seed(34234)
    exog = np.random.normal(size=(300, 4))
    exog[:, 0] = 1
    endog = np.dot(exog, np.r_[1, 1, 0, 0.2]) + np.random.normal(size=300)
    group = np.kron(np.arange(100), np.r_[1, 1, 1])
    vi = cov_struct.Independence()
    ve = cov_struct.Exchangeable()
    L = np.r_[[[0, 0, 0, 1]]]
    R = np.r_[0,]
    for j, v in enumerate((vi, ve)):
        md = gee.GEE(endog, exog, group, None, family, v, constraint=(L, R))
        mdf = md.fit()
        assert_almost_equal(mdf.params[3], 0, decimal=10)