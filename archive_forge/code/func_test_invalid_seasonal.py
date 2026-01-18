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
@pytest.mark.parametrize('seasonal', ('add', 'mul'))
def test_invalid_seasonal(seasonal):
    y = pd.Series(-np.ones(100), index=pd.date_range('2000-1-1', periods=100, freq='MS'))
    with pytest.raises(ValueError):
        ExponentialSmoothing(y, seasonal=seasonal, seasonal_periods=1)