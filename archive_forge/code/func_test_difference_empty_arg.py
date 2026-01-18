from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_difference_empty_arg(self, index, sort):
    first = index.copy()
    first = first[5:20]
    first.name = 'name'
    result = first.difference([], sort)
    expected = index[5:20].unique()
    expected.name = 'name'
    tm.assert_index_equal(result, expected)