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
def test_nominal(self):
    endog, exog, groups = load_data('gee_nominal_1.csv', icept=False)
    va = cov_struct.Independence()
    mod1 = gee.NominalGEE(endog, exog, groups, cov_struct=va)
    rslt1 = mod1.fit()
    cf1 = np.r_[0.450009, 0.451959, -0.918825, -0.468266]
    se1 = np.r_[0.08915936, 0.07005046, 0.12198139, 0.08281258]
    assert_allclose(rslt1.params, cf1, rtol=1e-05, atol=1e-05)
    assert_allclose(rslt1.standard_errors(), se1, rtol=1e-05, atol=1e-05)
    va = cov_struct.GlobalOddsRatio('nominal')
    mod2 = gee.NominalGEE(endog, exog, groups, cov_struct=va)
    rslt2 = mod2.fit(start_params=rslt1.params)
    cf2 = np.r_[0.455365, 0.415334, -0.916589, -0.502116]
    se2 = np.r_[0.08803614, 0.06628179, 0.12259726, 0.08411064]
    assert_allclose(rslt2.params, cf2, rtol=1e-05, atol=1e-05)
    assert_allclose(rslt2.standard_errors(), se2, rtol=1e-05, atol=1e-05)
    assert_equal(type(rslt1), gee.NominalGEEResultsWrapper)
    assert_equal(type(rslt1._results), gee.NominalGEEResults)