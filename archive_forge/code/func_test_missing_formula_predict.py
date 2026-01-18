from statsmodels.compat.python import lrange
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy.linalg import toeplitz
from scipy.stats import t as student_t
from statsmodels.datasets import longley
from statsmodels.regression.linear_model import (
from statsmodels.tools.tools import add_constant
def test_missing_formula_predict():
    nsample = 30
    data = np.linspace(0, 10, nsample)
    null = np.array([np.nan])
    data = pd.DataFrame({'x': np.concatenate((data, null))})
    beta = np.array([1, 0.1])
    e = np.random.normal(size=nsample + 1)
    data['y'] = beta[0] + beta[1] * data['x'] + e
    model = OLS.from_formula('y ~ x', data=data)
    fit = model.fit()
    fit.predict(exog=data[:-1])