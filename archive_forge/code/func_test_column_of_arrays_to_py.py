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
def test_column_of_arrays_to_py(self):
    dtype = 'i1'
    arr = np.array([np.arange(10, dtype=dtype), np.arange(5, dtype=dtype), None, np.arange(1, dtype=dtype)], dtype=object)
    type_ = pa.list_(pa.int8())
    parr = pa.array(arr, type=type_)
    assert parr[0].as_py() == list(range(10))
    assert parr[1].as_py() == list(range(5))
    assert parr[2].as_py() is None
    assert parr[3].as_py() == [0]