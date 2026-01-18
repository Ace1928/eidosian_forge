import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
import scipy.stats as stats
from statsmodels.discrete.discrete_model import Logit
from statsmodels.miscmodels.ordinal_model import OrderedModel
from statsmodels.tools.sm_exceptions import HessianInversionWarning
from statsmodels.tools.tools import add_constant
from .results.results_ordinal_model import data_store as ds
def test_loglikerelated(self):
    res1 = self.res1
    mod = res1.model
    fact = 1.1
    score1 = mod.score(res1.params * fact)
    score_obs_numdiff = mod.score_obs(res1.params * fact)
    score_obs_exog = mod.score_obs_(res1.params * fact)
    assert_allclose(score_obs_numdiff.sum(0), score1, atol=1e-06)
    assert_allclose(score_obs_exog.sum(0), score1[:mod.k_vars], atol=1e-06)
    mod_null = OrderedModel(mod.endog, None, offset=np.zeros(mod.nobs), distr=mod.distr)
    null_params = mod.start_params
    res_null = mod_null.fit(method='bfgs', disp=False)
    assert_allclose(res_null.params, null_params[mod.k_vars:], rtol=1e-08)
    assert_allclose(res1.llnull, res_null.llf, rtol=1e-08)