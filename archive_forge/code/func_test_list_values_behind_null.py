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
def test_list_values_behind_null(self):
    arr = pa.ListArray.from_arrays(offsets=pa.array([0, 2, 4, 6]), values=pa.array([1, 2, 99, 99, 3, None]), mask=pa.array([False, True, False]))
    np_arr = arr.to_numpy(zero_copy_only=False)
    expected = np.array([[1.0, 2.0], None, [3.0, np.nan]], dtype='object')
    for left, right in zip(np_arr, expected):
        if right is None:
            assert left == right
        else:
            npt.assert_array_equal(left, right)