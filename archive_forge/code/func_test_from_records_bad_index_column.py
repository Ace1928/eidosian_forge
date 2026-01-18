from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
import pytz
from pandas._config import using_pyarrow_string_dtype
from pandas.compat import is_platform_little_endian
from pandas import (
import pandas._testing as tm
def test_from_records_bad_index_column(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 3)), columns=['A', 'B', 'C'])
    with tm.assert_produces_warning(FutureWarning):
        df1 = DataFrame.from_records(df, index=['C'])
    tm.assert_index_equal(df1.index, Index(df.C))
    with tm.assert_produces_warning(FutureWarning):
        df1 = DataFrame.from_records(df, index='C')
    tm.assert_index_equal(df1.index, Index(df.C))
    msg = '|'.join(["'None of \\[2\\] are in the columns'"])
    with pytest.raises(KeyError, match=msg):
        with tm.assert_produces_warning(FutureWarning):
            DataFrame.from_records(df, index=[2])
    with pytest.raises(KeyError, match=msg):
        with tm.assert_produces_warning(FutureWarning):
            DataFrame.from_records(df, index=2)