from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
def moment_exponential_add(params, exog, exp=True):
    if not np.isfinite(params).all():
        print('invalid params', params)
    if exp:
        predicted = np.exp(np.dot(exog, params))
        predicted = np.clip(predicted, 0, 1e+100)
    else:
        predicted = np.dot(exog, params)
    return predicted