import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.testing import assert_series_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.compat.scipy import SP_LT_17
from statsmodels.datasets import cpunish, longley
from statsmodels.discrete import discrete_model as discrete
from statsmodels.genmod.generalized_linear_model import GLM, SET_USE_BIC_LLF
from statsmodels.tools.numdiff import (
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import add_constant
def test_invalid_endog_formula(self, reset_randomstate):
    n = 200
    exog = np.random.normal(size=(n, 2))
    endog = np.random.randint(0, 3, size=n).astype(str)
    data = pd.DataFrame({'y': endog, 'x1': exog[:, 0], 'x2': exog[:, 1]})
    with pytest.raises(ValueError, match='array with multiple columns'):
        sm.GLM.from_formula('y ~ x1 + x2', data, family=sm.families.Binomial())