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
def test_array_from_strided_numpy_array(self):
    np_arr = np.arange(0, 10, dtype=np.float32)[1:-1:2]
    pa_arr = pa.array(np_arr, type=pa.float64())
    expected = pa.array([1.0, 3.0, 5.0, 7.0], type=pa.float64())
    pa_arr.equals(expected)