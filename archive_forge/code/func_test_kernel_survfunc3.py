import os
import numpy as np
from statsmodels.duration.survfunc import (
from numpy.testing import assert_allclose
import pandas as pd
import pytest
@pytest.mark.smoke
def test_kernel_survfunc3():
    n = 100
    np.random.seed(3434)
    x = np.random.normal(size=(n, 3))
    time = np.random.randint(0, 10, size=n)
    status = np.random.randint(0, 2, size=n)
    SurvfuncRight(time, status, exog=x, bw_factor=10000)
    SurvfuncRight(time, status, exog=x, bw_factor=np.r_[10000, 10000])