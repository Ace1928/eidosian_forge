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
def test_glm_bic_warning(iris):
    X = np.c_[np.ones(100), iris[50:, :4]]
    y = np.array(iris)[50:, 4].astype(np.int32)
    y -= 1
    model = GLM(y, X, family=sm.families.Binomial()).fit()
    with pytest.warns(FutureWarning, match='The bic'):
        assert isinstance(model.bic, float)