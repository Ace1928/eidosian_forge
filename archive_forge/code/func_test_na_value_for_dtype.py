from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('dtype, na_value', [(np.dtype('M8[ns]'), np.datetime64('NaT', 'ns')), (np.dtype('m8[ns]'), np.timedelta64('NaT', 'ns')), (DatetimeTZDtype.construct_from_string('datetime64[ns, US/Eastern]'), NaT), (PeriodDtype('M'), NaT), ('u1', 0), ('u2', 0), ('u4', 0), ('u8', 0), ('i1', 0), ('i2', 0), ('i4', 0), ('i8', 0), ('bool', False), ('f2', np.nan), ('f4', np.nan), ('f8', np.nan), ('O', np.nan), (IntervalDtype(), np.nan)])
def test_na_value_for_dtype(dtype, na_value):
    result = na_value_for_dtype(pandas_dtype(dtype))
    assert result is na_value or (isna(result) and isna(na_value) and (type(result) is type(na_value)))