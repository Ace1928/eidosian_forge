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
def test_binary_pred_table_zeros():
    nobs = 10
    y = np.zeros(nobs)
    y[[1, 3]] = 1
    res = Logit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[8.0, 0.0], [2.0, 0.0]])
    assert_equal(res.pred_table(), expected)
    res = MNLogit(y, np.ones(nobs)).fit(disp=0)
    expected = np.array([[8.0, 0.0], [2.0, 0.0]])
    assert_equal(res.pred_table(), expected)