from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('swap_objs', [True, False])
def test_constructor_mixed_nat_objs_infers_object(self, swap_objs):
    data = [np.datetime64('nat'), np.timedelta64('nat')]
    if swap_objs:
        data = data[::-1]
    expected = Index(data, dtype=object)
    tm.assert_index_equal(Index(data), expected)
    tm.assert_index_equal(Index(np.array(data, dtype=object)), expected)