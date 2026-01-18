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
def test_array_from_py_float32():
    data = [[1.2, 3.4], [9.0, 42.0]]
    t = pa.float32()
    arr1 = pa.array(data[0], type=t)
    arr2 = pa.array(data, type=pa.list_(t))
    expected1 = np.array(data[0], dtype=np.float32)
    expected2 = pd.Series([np.array(data[0], dtype=np.float32), np.array(data[1], dtype=np.float32)])
    assert arr1.type == t
    assert arr1.equals(pa.array(expected1))
    assert arr2.equals(pa.array(expected2))