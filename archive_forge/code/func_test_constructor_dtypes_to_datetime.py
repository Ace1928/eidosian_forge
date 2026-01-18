from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('cast_index', [True, False])
@pytest.mark.parametrize('vals', [Index(np.array([np.datetime64('2011-01-01'), np.datetime64('2011-01-02')])), Index([datetime(2011, 1, 1), datetime(2011, 1, 2)])])
def test_constructor_dtypes_to_datetime(self, cast_index, vals):
    if cast_index:
        index = Index(vals, dtype=object)
        assert isinstance(index, Index)
        assert index.dtype == object
    else:
        index = Index(vals)
        assert isinstance(index, DatetimeIndex)