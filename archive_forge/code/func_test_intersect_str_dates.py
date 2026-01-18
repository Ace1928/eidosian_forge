from datetime import datetime
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas.core.dtypes.cast import find_common_type
from pandas import (
import pandas._testing as tm
from pandas.api.types import (
def test_intersect_str_dates(self):
    dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]
    index1 = Index(dt_dates, dtype=object)
    index2 = Index(['aa'], dtype=object)
    result = index2.intersection(index1)
    expected = Index([], dtype=object)
    tm.assert_index_equal(result, expected)