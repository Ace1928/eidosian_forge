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
def test_conversion_extensiontype_to_extensionarray(monkeypatch):
    import pandas.core.internals as _int
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(MyCustomIntegerType(), storage)
    table = pa.table({'a': arr})
    result = arr.to_pandas()
    assert isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)
    expected = pd.Series([1, 2, 3, 4], dtype='Int64')
    tm.assert_series_equal(result, expected)
    result = table.to_pandas()
    assert isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)
    expected = pd.DataFrame({'a': pd.array([1, 2, 3, 4], dtype='Int64')})
    tm.assert_frame_equal(result, expected)
    if Version(pd.__version__) < Version('1.3.0.dev'):
        monkeypatch.delattr(pd.core.arrays.integer._IntegerDtype, '__from_arrow__')
    else:
        monkeypatch.delattr(pd.core.arrays.integer.NumericDtype, '__from_arrow__')
    result = arr.to_pandas()
    assert not isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)
    expected = pd.Series([1, 2, 3, 4])
    tm.assert_series_equal(result, expected)
    with pytest.raises(ValueError):
        table.to_pandas()