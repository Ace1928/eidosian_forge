from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def check_jac(self, res=None):
    if res is None:
        res1 = self.res1
    else:
        res1 = res
    exog = res1.model.exog
    jacsum = res1.model.score_obs(res1.params).sum(0)
    score = res1.model.score(res1.params)
    assert_almost_equal(jacsum, score, DECIMAL_9)
    if isinstance(res1.model, (NegativeBinomial, MNLogit)):
        return
    s1 = res1.model.score_obs(res1.params)
    sf = res1.model.score_factor(res1.params)
    if not isinstance(sf, tuple):
        s2 = sf[:, None] * exog
    else:
        sf0, sf1 = sf
        s2 = np.column_stack((sf0[:, None] * exog, sf1))
    assert_allclose(s2, s1, rtol=1e-10)
    h1 = res1.model.hessian(res1.params)
    hf = res1.model.hessian_factor(res1.params)
    if not isinstance(hf, tuple):
        h2 = (hf * exog.T).dot(exog)
    else:
        hf0, hf1, hf2 = hf
        h00 = (hf0 * exog.T).dot(exog)
        h10 = np.atleast_2d(hf1.T.dot(exog))
        h11 = np.atleast_2d(hf2.sum(0))
        h2 = np.vstack((np.column_stack((h00, h10.T)), np.column_stack((h10, h11))))
    assert_allclose(h2, h1, rtol=1e-10)