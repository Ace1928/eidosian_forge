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
def test_cov_confint_pandas():
    data = sm.datasets.anes96.load_pandas()
    exog = sm.add_constant(data.exog, prepend=False)
    res1 = sm.MNLogit(data.endog, exog).fit(method='newton', disp=0)
    cov = res1.cov_params()
    ci = res1.conf_int()
    se = np.sqrt(np.diag(cov))
    se2 = (ci.iloc[:, 1] - ci.iloc[:, 0]) / (2 * stats.norm.ppf(0.975))
    assert_allclose(se, se2)
    assert_index_equal(ci.index, cov.index)
    assert_index_equal(cov.index, cov.columns)
    assert isinstance(ci.index, pd.MultiIndex)