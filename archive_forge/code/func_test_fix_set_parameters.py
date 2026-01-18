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
def test_fix_set_parameters(ses):
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, initial_level=1.0, initialization_method='heuristic')
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, trend='add', initial_trend=1.0, initialization_method='legacy-heuristic')
    with pytest.raises(ValueError):
        ExponentialSmoothing(ses, seasonal='add', initial_seasonal=np.ones(12), initialization_method='estimated')