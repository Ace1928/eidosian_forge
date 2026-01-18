from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype', ['int64', 'uint64'])
def test_constructor_int_dtype_nan_raises(self, dtype):
    data = [np.nan]
    msg = 'cannot convert'
    with pytest.raises(ValueError, match=msg):
        Index(data, dtype=dtype)