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
def test_damping_trend_zero():
    endog = np.arange(10)
    mod = ExponentialSmoothing(endog, trend='add', damped_trend=True, initialization_method='estimated')
    res1 = mod.fit(smoothing_level=1, smoothing_trend=0.0, damping_trend=1e-20)
    pred1 = res1.predict(start=0)
    assert_allclose(pred1, np.r_[0.0, np.arange(9)], atol=1e-10)
    res2 = mod.fit(smoothing_level=1, smoothing_trend=0.0, damping_trend=0)
    pred2 = res2.predict(start=0)
    assert_allclose(pred2, np.r_[0.0, np.arange(9)], atol=1e-10)
    assert_allclose(pred1, pred2, atol=1e-10)