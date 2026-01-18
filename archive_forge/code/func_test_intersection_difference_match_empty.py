from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_intersection_difference_match_empty(self, index, sort):
    if not index.is_unique:
        pytest.skip('Not relevant because index is not unique')
    inter = index.intersection(index[:0])
    diff = index.difference(index, sort=sort)
    tm.assert_index_equal(inter, diff, exact=True)