from statsmodels.compat.pandas import (
from statsmodels.compat.pytest import pytest_warns
from collections.abc import Hashable
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.deterministic import (
@pytest.mark.parametrize('index', [pd.RangeIndex(0, 200), pd.Index(np.arange(200)), pd.date_range('2000-1-1', freq='MS', periods=200), pd.period_range('2000-1-1', freq='M', periods=200)])
def test_determintic_term_equiv(index):
    base = DeterministicProcess(pd.RangeIndex(0, 200), constant=True, order=2)
    dp = DeterministicProcess(index, constant=True, order=2)
    np.testing.assert_array_equal(base.in_sample(), dp.in_sample())
    np.testing.assert_array_equal(base.out_of_sample(37), dp.out_of_sample(37))
    np.testing.assert_array_equal(base.range(200, 237), dp.range(200, 237))
    np.testing.assert_array_equal(base.range(50, 150), dp.range(50, 150))
    np.testing.assert_array_equal(base.range(50, 250), dp.range(50, 250))