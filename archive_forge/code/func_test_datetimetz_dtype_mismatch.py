import numpy as np
import pytest
from pandas.core.dtypes.cast import find_common_type
from pandas.core.dtypes.common import pandas_dtype
from pandas.core.dtypes.dtypes import (
from pandas import (
@pytest.mark.parametrize('dtype2', [DatetimeTZDtype(unit='ns', tz='Asia/Tokyo'), np.dtype('datetime64[ns]'), object, np.int64])
def test_datetimetz_dtype_mismatch(dtype2):
    dtype = DatetimeTZDtype(unit='ns', tz='US/Eastern')
    assert find_common_type([dtype, dtype2]) == object
    assert find_common_type([dtype2, dtype]) == object