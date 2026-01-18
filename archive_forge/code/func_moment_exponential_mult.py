from statsmodels.compat.python import lmap
import numpy as np
import pandas
from scipy import stats
import pytest
from statsmodels.regression.linear_model import OLS
from statsmodels.sandbox.regression import gmm
from numpy.testing import assert_allclose, assert_equal
def moment_exponential_mult(params, data, exp=True):
    endog = data[:, 0]
    exog = data[:, 1:]
    if not np.isfinite(params).all():
        print('invalid params', params)
    if exp:
        predicted = np.exp(np.dot(exog, params))
        predicted = np.clip(predicted, 0, 1e+100)
        resid = endog / predicted - 1
        if not np.isfinite(resid).all():
            print('invalid resid', resid)
    else:
        resid = endog - np.dot(exog, params)
    return resid