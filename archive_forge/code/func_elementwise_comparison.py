import operator
import numpy as np
import pytest
from pandas.core.dtypes.common import is_list_like
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
from pandas.tests.arithmetic.common import get_upcast_box
def elementwise_comparison(self, op, interval_array, other):
    """
        Helper that performs elementwise comparisons between `array` and `other`
        """
    other = other if is_list_like(other) else [other] * len(interval_array)
    expected = np.array([op(x, y) for x, y in zip(interval_array, other)])
    if isinstance(other, Series):
        return Series(expected, index=other.index)
    return expected