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
def test_basic_inference(self):
    res1 = self.res1
    res2 = self.res2
    rtol = 1e-07
    assert_allclose(res1.params, res2.params, rtol=1e-08)
    assert_allclose(res1.bse, res2.bse, rtol=rtol)
    assert_allclose(res1.tvalues, res2.tvalues, rtol=rtol, atol=1e-08)
    assert_allclose(res1.pvalues, res2.pvalues, rtol=rtol, atol=1e-20)
    ci = res2.params_table[:, 4:6]
    assert_allclose(res1.conf_int(), ci, rtol=5e-07, atol=1e-20)