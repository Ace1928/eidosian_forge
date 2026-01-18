import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
def test_cast_timestamp_unit():
    val = datetime.now()
    s = pd.Series([val])
    s_nyc = s.dt.tz_localize('tzlocal()').dt.tz_convert('America/New_York')
    us_with_tz = pa.timestamp('us', tz='America/New_York')
    arr = pa.Array.from_pandas(s_nyc, type=us_with_tz)
    assert arr.type == us_with_tz
    arr2 = pa.Array.from_pandas(s, type=pa.timestamp('us'))
    assert arr[0].as_py() == s_nyc[0].to_pydatetime()
    assert arr2[0].as_py() == s[0].to_pydatetime()
    arr = pa.array([123123], type='int64').cast(pa.timestamp('ms'))
    expected = pa.array([123], type='int64').cast(pa.timestamp('s'))
    assert arr.type == pa.timestamp('ms')
    target = pa.timestamp('s')
    with pytest.raises(ValueError):
        arr.cast(target)
    result = arr.cast(target, safe=False)
    assert result.equals(expected)
    series = pd.Series([pd.Timestamp(1), pd.Timestamp(10), pd.Timestamp(1000)])
    expected = pa.array([0, 0, 1], type=pa.timestamp('us'))
    with pytest.raises(ValueError):
        pa.array(series, type=pa.timestamp('us'))
    with pytest.raises(ValueError):
        pa.Array.from_pandas(series, type=pa.timestamp('us'))
    result = pa.Array.from_pandas(series, type=pa.timestamp('us'), safe=False)
    assert result.equals(expected)
    result = pa.array(series, type=pa.timestamp('us'), safe=False)
    assert result.equals(expected)