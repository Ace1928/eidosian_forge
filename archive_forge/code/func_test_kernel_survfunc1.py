import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
def test_kernel_survfunc1():
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.uniform(size=n)
    status = np.random.randint(0, 2, size=n)
    result = SurvfuncRight(time, status, exog=x)
    timex = np.r_[0.30721103, 0.0515439, 0.69246897, 0.16446079, 0.31308528]
    sprob = np.r_[0.98948277, 0.98162275, 0.97129237, 0.96044668, 0.95030368]
    assert_allclose(result.time[0:5], timex)
    assert_allclose(result.surv_prob[0:5], sprob)