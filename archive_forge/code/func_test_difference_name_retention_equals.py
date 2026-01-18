from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_name_retention_equals(self, index, names):
    if isinstance(index, MultiIndex):
        names = [[x] * index.nlevels for x in names]
    index = index.rename(names[0])
    other = index.rename(names[1])
    assert index.equals(other)
    result = index.difference(other)
    expected = index[:0].rename(names[2])
    tm.assert_index_equal(result, expected)