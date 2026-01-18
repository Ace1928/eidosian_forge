from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_different_inputs():
    array_input_add = [10, 20, 30, 40, 50]
    series_index_add = pd.date_range(start='2000-1-1', periods=len(array_input_add))
    series_input_add = pd.Series(array_input_add, series_index_add)
    array_input_mul = [2, 4, 8, 16, 32]
    series_index_mul = pd.date_range(start='2000-1-1', periods=len(array_input_mul))
    series_input_mul = pd.Series(array_input_mul, series_index_mul)
    fit1 = ExponentialSmoothing(array_input_add, trend='add').fit()
    fit2 = ExponentialSmoothing(series_input_add, trend='add').fit()
    fit3 = ExponentialSmoothing(array_input_mul, trend='mul').fit()
    fit4 = ExponentialSmoothing(series_input_mul, trend='mul').fit()
    assert_almost_equal(fit1.predict(), [60], 1)
    assert_almost_equal(fit1.predict(start=5, end=7), [60, 70, 80], 1)
    assert_almost_equal(fit2.predict(), [60], 1)
    assert_almost_equal(fit2.predict(start='2000-1-6', end='2000-1-8'), [60, 70, 80], 1)
    assert_almost_equal(fit3.predict(), [64], 1)
    assert_almost_equal(fit3.predict(start=5, end=7), [64, 128, 256], 1)
    assert_almost_equal(fit4.predict(), [64], 1)
    assert_almost_equal(fit4.predict(start='2000-1-6', end='2000-1-8'), [64, 128, 256], 1)