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
def test_glm_start_params():
    y2 = np.array('0 1 0 0 0 1'.split(), int)
    wt = np.array([50, 1, 50, 1, 5, 10])
    y2 = np.repeat(y2, wt)
    x2 = np.repeat([0, 0, 0.001, 100, -1, -1], wt)
    mod = sm.GLM(y2, sm.add_constant(x2), family=sm.families.Binomial())
    res = mod.fit(start_params=[-4, -5])
    np.testing.assert_almost_equal(res.params, [-4.60305022, -5.29634545], 6)