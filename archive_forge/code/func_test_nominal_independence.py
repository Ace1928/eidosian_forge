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
@pytest.mark.smoke
def test_nominal_independence(self):
    np.random.seed(434)
    n = 40
    y = np.random.randint(0, 3, n)
    groups = np.kron(np.arange(n / 2), np.r_[1, 1])
    x = np.random.normal(size=(n, 1))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        nmi = cov_struct.NominalIndependence()
        model1 = gee.NominalGEE(y, x, groups, cov_struct=nmi)
        model1.fit()