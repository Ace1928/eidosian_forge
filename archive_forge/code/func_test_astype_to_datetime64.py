import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype, expected', [('datetime64[ns]', np.array(['2015-01-01T00:00:00.000000000'], dtype='datetime64[ns]')), ('datetime64[ns, MET]', DatetimeIndex([Timestamp('2015-01-01 00:00:00+0100', tz='MET')]).array)])
def test_astype_to_datetime64(self, dtype, expected):
    result = Categorical(['2015-01-01']).astype(dtype)
    assert result == expected