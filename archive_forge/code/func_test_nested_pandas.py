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
def test_nested_pandas(self):
    np.random.seed(4234)
    n = 10000
    groups = np.kron(np.arange(n // 100), np.ones(100)).astype(int)
    groups1 = np.kron(np.arange(n // 50), np.ones(50)).astype(int)
    groups2 = np.kron(np.arange(n // 10), np.ones(10)).astype(int)
    groups_e = np.random.normal(size=n // 100)
    groups1_e = 2 * np.random.normal(size=n // 50)
    groups2_e = 3 * np.random.normal(size=n // 10)
    y = groups_e[groups] + groups1_e[groups1] + groups2_e[groups2]
    y += 0.5 * np.random.normal(size=n)
    df = pd.DataFrame({'y': y, 'TheGroups': groups, 'groups1': groups1, 'groups2': groups2})
    model = gee.GEE.from_formula('y ~ 1', groups='TheGroups', dep_data='0 + groups1 + groups2', cov_struct=cov_struct.Nested(), data=df)
    result = model.fit()
    smry = result.cov_struct.summary()
    assert_allclose(smry.Variance, np.r_[1.437299, 4.421543, 8.905295, 0.25848], atol=1e-05, rtol=1e-05)