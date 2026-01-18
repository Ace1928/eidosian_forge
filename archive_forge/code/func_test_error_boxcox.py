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
def test_error_boxcox():
    y = np.random.standard_normal(100)
    with pytest.raises(TypeError, match='use_boxcox must be True'):
        ExponentialSmoothing(y, use_boxcox='a', initialization_method='known')
    mod = ExponentialSmoothing(y ** 2, use_boxcox=True)
    assert isinstance(mod, ExponentialSmoothing)
    mod = ExponentialSmoothing(y ** 2, use_boxcox=True, initialization_method='legacy-heuristic')
    with pytest.raises(ValueError, match='use_boxcox was set'):
        mod.fit(use_boxcox=False)