import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_raises
import pandas as pd
import pytest
import statsmodels.api as sm
from statsmodels.datasets.cpunish import load
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.tools.sm_exceptions import SpecificationWarning
from statsmodels.tools.tools import add_constant
from .results import (
def test_incompatible_input():
    weights = [1, 1, 1, 2, 2, 2, 3, 3, 3, 1, 1, 1, 2, 2, 2, 3, 3]
    exog = cpunish_data.exog
    endog = cpunish_data.endog
    family = sm.families.Poisson()
    assert_raises(ValueError, GLM, endog, exog, family=family, freq_weights=weights[:-1])
    assert_raises(ValueError, GLM, endog, exog, family=family, var_weights=weights[:-1])
    assert_raises(ValueError, GLM, endog, exog, family=family, freq_weights=weights + [3])
    assert_raises(ValueError, GLM, endog, exog, family=family, var_weights=weights + [3])
    assert_raises(ValueError, GLM, endog, exog, family=family, freq_weights=[weights, weights])
    assert_raises(ValueError, GLM, endog, exog, family=family, var_weights=[weights, weights])