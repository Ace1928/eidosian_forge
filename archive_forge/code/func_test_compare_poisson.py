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
def test_compare_poisson(self):
    vs = cov_struct.Independence()
    family = families.Poisson()
    np.random.seed(34234)
    Y = np.ceil(-np.log(np.random.uniform(size=100)))
    X1 = np.random.normal(size=100)
    X2 = np.random.normal(size=100)
    X3 = np.random.normal(size=100)
    groups = np.random.randint(0, 4, size=100)
    D = pd.DataFrame({'Y': Y, 'X1': X1, 'X2': X2, 'X3': X3})
    mod1 = gee.GEE.from_formula('Y ~ X1 + X2 + X3', groups, D, family=family, cov_struct=vs)
    rslt1 = mod1.fit()
    mod2 = discrete.Poisson.from_formula('Y ~ X1 + X2 + X3', data=D)
    rslt2 = mod2.fit(disp=False)
    assert_almost_equal(rslt1.params.values, rslt2.params.values, decimal=10)