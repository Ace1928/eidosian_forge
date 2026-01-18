from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
def test_invalid_formcast_index(index):
    tt = TimeTrend(order=4)
    with pytest.raises(ValueError, match='The number of values in forecast_'):
        tt.out_of_sample(10, index, pd.RangeIndex(11))