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
@pytest.mark.parametrize('trend_seasonal', (('mul', None), (None, 'mul'), ('mul', 'mul')))
def test_negative_multipliative(trend_seasonal):
    trend, seasonal = trend_seasonal
    y = -np.ones(100)
    with pytest.raises(ValueError):
        ExponentialSmoothing(y, trend=trend, seasonal=seasonal, seasonal_periods=10)