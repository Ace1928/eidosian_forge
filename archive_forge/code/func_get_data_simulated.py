import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pytest
from statsmodels.tools.tools import add_constant
from statsmodels.base._prediction_inference import PredictionResultsMonotonic
from statsmodels.discrete.discrete_model import (
from statsmodels.discrete.count_model import (
from statsmodels.sandbox.regression.tests.test_gmm_poisson import DATA
from .results import results_predict as resp
def get_data_simulated():
    np.random.seed(987456348)
    nobs = 500
    x = np.ones((nobs, 1))
    yn = np.random.randn(nobs)
    y = 1 * (1.5 + yn) ** 2
    y = np.trunc(y + 0.5)
    return (y, x)