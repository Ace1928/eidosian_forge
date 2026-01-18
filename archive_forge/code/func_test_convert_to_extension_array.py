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
def test_convert_to_extension_array(monkeypatch):
    import pandas.core.internals as _int
    df = pd.DataFrame({'a': [1, 2, 3], 'b': pd.array([2, 3, 4], dtype='Int64'), 'c': [4, 5, 6]})
    table = pa.table(df)
    result = table.to_pandas()
    assert not isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)
    assert _get_mgr(result).blocks[0].values.dtype == np.dtype('int64')
    assert isinstance(_get_mgr(result).blocks[1], _int.ExtensionBlock)
    tm.assert_frame_equal(result, df)
    df2 = pd.DataFrame({'a': pd.array([1, 2, None], dtype='Int64')})
    table2 = pa.table(df2)
    result = table2.to_pandas()
    assert isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)
    tm.assert_frame_equal(result, df2)
    if Version(pd.__version__) < Version('1.3.0.dev'):
        monkeypatch.delattr(pd.core.arrays.integer._IntegerDtype, '__from_arrow__')
    else:
        monkeypatch.delattr(pd.core.arrays.integer.NumericDtype, '__from_arrow__')
    result = table.to_pandas()
    assert len(_get_mgr(result).blocks) == 1
    assert not isinstance(_get_mgr(result).blocks[0], _int.ExtensionBlock)