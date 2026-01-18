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
def test_hw_seasonal_add_mul(self):
    mod2 = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='add', seasonal='mul', initialization_method='estimated', use_boxcox=True)
    fit2 = mod2.fit()
    assert_almost_equal(fit2.forecast(8), [61.69, 37.37, 47.22, 52.03, 65.08, 39.34, 49.72, 54.79], 2)
    ExponentialSmoothing(self.aust, seasonal_periods=4, trend='mul', seasonal='add', initialization_method='estimated', use_boxcox=0.0).fit()
    ExponentialSmoothing(self.aust, seasonal_periods=4, trend='multiplicative', seasonal='multiplicative', initialization_method='estimated', use_boxcox=0.0).fit()