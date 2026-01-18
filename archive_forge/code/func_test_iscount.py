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
def test_iscount():
    X = np.random.random((50, 10))
    X[:, 2] = np.random.randint(1, 10, size=50)
    X[:, 6] = np.random.randint(1, 10, size=50)
    X[:, 4] = np.random.randint(0, 2, size=50)
    X[:, 1] = np.random.randint(-10, 10, size=50)
    count_ind = _iscount(X)
    assert_equal(count_ind, [2, 6])