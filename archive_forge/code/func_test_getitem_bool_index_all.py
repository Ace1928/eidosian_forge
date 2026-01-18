from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('ind1', [[True] * 5, Index([True] * 5)])
@pytest.mark.parametrize('ind2', [[True, False, True, False, False], Index([True, False, True, False, False])])
def test_getitem_bool_index_all(ind1, ind2):
    idx = MultiIndex.from_tuples([(10, 1), (20, 2), (30, 3), (40, 4), (50, 5)])
    tm.assert_index_equal(idx[ind1], idx)
    expected = MultiIndex.from_tuples([(10, 1), (30, 3)])
    tm.assert_index_equal(idx[ind2], expected)