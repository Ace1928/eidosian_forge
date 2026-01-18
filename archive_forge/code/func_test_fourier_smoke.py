from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.smoke
def test_fourier_smoke(index, forecast_index):
    f = Fourier(12, 2)
    f.in_sample(index)
    steps = 83 if forecast_index is None else len(forecast_index)
    warn = None
    if is_int_index(index) and np.any(np.diff(index) != 1) or (type(index) is pd.Index and max(index) > 2 ** 63 and (forecast_index is None)):
        warn = UserWarning
    with pytest_warns(warn):
        f.out_of_sample(steps, index, forecast_index)
    assert isinstance(f.period, float)
    assert isinstance(f.order, int)
    str(f)
    hash(f)
    with pytest.raises(ValueError, match='2 \\* order must be <= period'):
        Fourier(12, 7)