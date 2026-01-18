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
@pytest.mark.parametrize('trend', TRENDS)
@pytest.mark.parametrize('seasonal', SEASONALS)
def test_equivalence_cython_python(trend, seasonal):
    mod = ExponentialSmoothing(housing_data, trend=trend, seasonal=seasonal, initialization_method='estimated')
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    params = res.params
    nobs = housing_data.shape[0]
    y = np.squeeze(np.asarray(housing_data))
    m = 12 if seasonal else 0
    p = np.zeros(6 + m)
    alpha = params['smoothing_level']
    beta = params['smoothing_trend']
    gamma = params['smoothing_seasonal']
    phi = params['damping_trend']
    phi = 1.0 if np.isnan(phi) else phi
    l0 = params['initial_level']
    b0 = params['initial_trend']
    p[:6] = (alpha, beta, gamma, l0, b0, phi)
    if seasonal:
        p[6:] = params['initial_seasons']
    xi = np.ones_like(p).astype(np.int64)
    p_copy = p.copy()
    bounds = np.array([[0.0, 1.0]] * 3)
    py_func = PY_SMOOTHERS[seasonal, trend]
    cy_func = SMOOTHERS[seasonal, trend]
    py_hw_args = PyHoltWintersArgs(xi, p_copy, bounds, y, m, nobs, False)
    cy_hw_args = HoltWintersArgs(xi, p_copy, bounds, y, m, nobs, False)
    sse_cy = cy_func(p, cy_hw_args)
    sse_py = py_func(p, py_hw_args)
    assert_allclose(sse_py, sse_cy)
    sse_py = py_func(p, cy_hw_args)
    assert_allclose(sse_py, sse_cy)