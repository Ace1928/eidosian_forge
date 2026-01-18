from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import assert_equal, assert_, assert_raises
import pandas as pd
import pytest
from statsmodels.base import data as sm_data
from statsmodels.formula import handle_formula_data
from statsmodels.regression.linear_model import OLS
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from statsmodels.discrete.discrete_model import Logit
def test_mv_endog(self):
    y = self.X
    y = y.loc[~np.isnan(y.values).any(axis=1)]
    data = sm_data.handle_data(self.X, None, 'drop')
    np.testing.assert_array_equal(data.endog, y.values)