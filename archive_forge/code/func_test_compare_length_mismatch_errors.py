import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
@pytest.mark.parametrize('length', [1, 3, 5])
@pytest.mark.parametrize('other_constructor', [IntervalArray, list])
def test_compare_length_mismatch_errors(self, op, other_constructor, length):
    interval_array = IntervalArray.from_arrays(range(4), range(1, 5))
    other = other_constructor([Interval(0, 1)] * length)
    with pytest.raises(ValueError, match='Lengths must match to compare'):
        op(interval_array, other)