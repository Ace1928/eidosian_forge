import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_kernel_survfunc2():
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(0, 10, size=n)
    status = np.random.randint(0, 2, size=n)
    resultkm = SurvfuncRight(time, status)
    result = SurvfuncRight(time, status, exog=x, bw_factor=10000)
    assert_allclose(resultkm.surv_times, result.surv_times)
    assert_allclose(resultkm.surv_prob, result.surv_prob, rtol=1e-06, atol=1e-06)