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
def test_nested_smaller_ints(self):
    data = pd.Series([np.array([1, 2, 3], dtype='i1'), None])
    result = pa.array(data)
    result2 = pa.array(data.values)
    expected = pa.array([[1, 2, 3], None], type=pa.list_(pa.int8()))
    assert result.equals(expected)
    assert result2.equals(expected)
    data3 = pd.Series([np.array([1, 2, 3], dtype='f4'), None])
    result3 = pa.array(data3)
    expected3 = pa.array([[1, 2, 3], None], type=pa.list_(pa.float32()))
    assert result3.equals(expected3)