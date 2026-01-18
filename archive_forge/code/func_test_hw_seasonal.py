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
def test_hw_seasonal(self):
    mod = ExponentialSmoothing(self.aust, seasonal_periods=4, trend='additive', seasonal='additive', initialization_method='estimated', use_boxcox=True)
    fit1 = mod.fit()
    assert_almost_equal(fit1.forecast(8), [59.96, 38.63, 47.48, 51.89, 62.81, 41.0, 50.06, 54.57], 2)