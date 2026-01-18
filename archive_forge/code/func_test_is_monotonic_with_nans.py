import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('attr', ['is_monotonic_increasing', 'is_monotonic_decreasing'])
@pytest.mark.parametrize('values', [[(np.nan,), (1,), (2,)], [(1,), (np.nan,), (2,)], [(1,), (2,), (np.nan,)]])
def test_is_monotonic_with_nans(values, attr):
    idx = MultiIndex.from_tuples(values, names=['test'])
    assert getattr(idx, attr) is False