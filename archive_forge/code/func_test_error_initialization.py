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
def test_error_initialization(ses):
    with pytest.raises(ValueError, match="initialization is 'known' but initial_level"):
        ExponentialSmoothing(ses, initialization_method='known')
    with pytest.raises(ValueError, match='initial_trend set but model'):
        ExponentialSmoothing(ses, initialization_method='known', initial_level=1.0, initial_trend=1.0)
    with pytest.raises(ValueError, match='initial_seasonal set but model'):
        ExponentialSmoothing(ses, initialization_method='known', initial_level=1.0, initial_seasonal=[0.2, 0.3, 0.4, 0.5])
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_level=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_trend=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_seasonal=[1.0, 0.2, 0.05, 4])
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, trend='add', initialization_method='known', initial_level=1.0)
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, trend='add', seasonal='add', initialization_method='known', initial_level=1.0, initial_trend=2.0)
    mod = ExponentialSmoothing(ses, initialization_method='known', initial_level=1.0)
    with pytest.raises(ValueError):
        mod.fit(initial_level=2.0)
    with pytest.raises(ValueError):
        mod.fit(use_basinhopping=True, method='least_squares')