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
def test_integer_byteorder(self):
    int_dtypes = ['i1', 'i2', 'i4', 'i8', 'u1', 'u2', 'u4', 'u8']
    for dt in int_dtypes:
        for order in '=<>':
            data = np.array([1, 2, 42], dtype=order + dt)
            for np_arr in (data, data[::2]):
                if data.dtype.isnative:
                    arr = pa.array(data)
                    assert arr.to_pylist() == data.tolist()
                else:
                    with pytest.raises(NotImplementedError):
                        arr = pa.array(data)