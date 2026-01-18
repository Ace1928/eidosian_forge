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
def test_margeff(self):
    if isinstance(self.res2.model, OLS) or hasattr(self.res1.model, 'offset'):
        pytest.skip('not available yet')
    marg1 = self.res1.get_margeff()
    marg2 = self.res2.get_margeff()
    assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
    assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)
    marg1 = self.res1.get_margeff(count=True, dummy=True)
    marg2 = self.res2.get_margeff(count=True, dummy=True)
    assert_allclose(marg1.margeff, marg2.margeff, rtol=1e-10)
    assert_allclose(marg1.margeff_se, marg2.margeff_se, rtol=1e-10)