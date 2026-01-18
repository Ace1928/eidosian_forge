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
def test_invalid_endog(self, reset_randomstate):
    endog = np.random.randint(0, 100, size=(1000, 3))
    exog = np.random.standard_normal((1000, 2))
    with pytest.raises(ValueError, match='endog has more than 2 columns'):
        GLM(endog, exog, family=sm.families.Binomial())