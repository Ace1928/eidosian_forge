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
@pytest.mark.parametrize('ix', [10, 100, 1000, 2000])
def test_forecast_index(ix):
    ts_1 = pd.Series([85601, 89662, 85122, 84400, 78250, 84434, 71072, 70357, 72635, 73210], index=range(ix, ix + 10))
    with pytest.warns(ConvergenceWarning):
        model = ExponentialSmoothing(ts_1, trend='add', damped_trend=False).fit()
    index = model.forecast(steps=10).index
    assert index[0] == ix + 10
    assert index[-1] == ix + 19