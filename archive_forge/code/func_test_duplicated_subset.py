import re
import sys
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('subset', [None, ['A', 'B'], 'A'])
def test_duplicated_subset(subset, keep):
    df = DataFrame({'A': [0, 1, 1, 2, 0], 'B': ['a', 'b', 'b', 'c', 'a'], 'C': [np.nan, 3, 3, None, np.nan]})
    if subset is None:
        subset = list(df.columns)
    elif isinstance(subset, str):
        subset = [subset]
    expected = df[subset].duplicated(keep=keep)
    result = df.duplicated(keep=keep, subset=subset)
    tm.assert_series_equal(result, expected)