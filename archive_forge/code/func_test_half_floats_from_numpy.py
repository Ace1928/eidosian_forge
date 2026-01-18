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
def test_half_floats_from_numpy(self):
    arr = np.array([1.5, np.nan], dtype=np.float16)
    a = pa.array(arr, type=pa.float16())
    x, y = a.to_pylist()
    assert isinstance(x, np.float16)
    assert x == 1.5
    assert isinstance(y, np.float16)
    assert np.isnan(y)
    a = pa.array(arr, type=pa.float16(), from_pandas=True)
    x, y = a.to_pylist()
    assert isinstance(x, np.float16)
    assert x == 1.5
    assert y is None