from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_should_not_compare(self):
    left = Index([1, 1])
    right = Index([True])
    result = left.difference(right)
    expected = Index([1])
    tm.assert_index_equal(result, expected)