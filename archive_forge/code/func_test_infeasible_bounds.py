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
def test_infeasible_bounds(ses):
    bounds = {'smoothing_level': (0.1, 0.2), 'smoothing_trend': (0.3, 0.4)}
    with pytest.raises(ValueError, match='The bounds for smoothing_trend'):
        ExponentialSmoothing(ses, trend='add', bounds=bounds, initialization_method='estimated').fit()
    bounds = {'smoothing_level': (0.3, 0.5), 'smoothing_seasonal': (0.7, 0.8)}
    with pytest.raises(ValueError, match='The bounds for smoothing_seasonal'):
        ExponentialSmoothing(ses, seasonal='add', bounds=bounds, initialization_method='estimated').fit()