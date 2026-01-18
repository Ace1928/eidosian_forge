from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('na_value', [None, np.nan])
@pytest.mark.parametrize('vtype', [list, tuple, iter])
def test_construction_list_tuples_nan(self, na_value, vtype):
    values = [(1, 'two'), (3.0, na_value)]
    result = Index(vtype(values))
    expected = MultiIndex.from_tuples(values)
    tm.assert_index_equal(result, expected)