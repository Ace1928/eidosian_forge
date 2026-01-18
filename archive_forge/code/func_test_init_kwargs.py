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
def test_init_kwargs(self):
    endog = self.res1.model.endog
    exog = self.res1.model.exog
    z = np.ones(len(endog))
    with pytest.warns(ValueWarning, match='unknown kwargs'):
        Probit(endog, exog, weights=z)