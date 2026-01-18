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
@pytest.mark.parametrize('trend', [None, 'add'])
@pytest.mark.parametrize('seasonal', [None, 'add'])
@pytest.mark.parametrize('nobs', [9, 10])
def test_estimated_initialization_short_data(ses, trend, seasonal, nobs):
    res = ExponentialSmoothing(ses[:nobs], trend=trend, seasonal=seasonal, seasonal_periods=4, initialization_method='estimated').fit()
    assert res.mle_retvals.success