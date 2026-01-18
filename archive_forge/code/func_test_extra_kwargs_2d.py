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
def test_extra_kwargs_2d(self):
    sigma = np.random.random((25, 25))
    sigma = sigma + sigma.T - np.diag(np.diag(sigma))
    data = sm_data.handle_data(self.y, self.X, 'drop', sigma=sigma)
    idx = ~np.isnan(np.c_[self.y, self.X]).any(axis=1)
    sigma = sigma[idx][:, idx]
    np.testing.assert_array_equal(data.sigma, sigma)