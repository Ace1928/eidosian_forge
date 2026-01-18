import os
import numpy as np
import pandas as pd
import pytest
import statsmodels.discrete.discrete_model as smd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.genmod.families import links
from statsmodels.regression.linear_model import OLS
from statsmodels.base.covtype import get_robustcov_results
import statsmodels.stats.sandwich_covariance as sw
from statsmodels.tools.tools import add_constant
from numpy.testing import assert_allclose, assert_equal, assert_
import statsmodels.tools._testing as smt
from .results import results_count_robust_cluster as results_st
def test_cov_options(self):
    kwdsa = {'weights_func': sw.weights_uniform, 'maxlags': 2}
    res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsa)
    res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsa)
    assert_allclose(res1a.bse, self.res1.bse, rtol=1e-12)
    assert_allclose(res2a.bse, self.res2.bse, rtol=1e-12)
    bse = np.array([2.82203924, 4.60199596, 11.01275064])
    assert_allclose(res1a.bse, bse, rtol=1e-06)
    assert_(res1a.cov_kwds['weights_func'] is sw.weights_uniform)
    kwdsb = {'kernel': sw.weights_bartlett, 'maxlags': 2}
    res1a = self.res1.model.fit(cov_type='HAC', cov_kwds=kwdsb)
    res2a = self.res2.model.fit(cov_type='HAC', cov_kwds=kwdsb)
    assert_allclose(res1a.bse, res2a.bse, rtol=1e-12)
    bse = np.array([2.502264, 3.697807, 9.193303])
    assert_allclose(res1a.bse, bse, rtol=1e-06)