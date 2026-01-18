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
def test_list_of_dictionary(self):
    child = pa.array(['foo', 'bar', None, 'foo']).dictionary_encode()
    arr = pa.ListArray.from_arrays([0, 1, 3, 3, 4], child)
    expected = pd.Series(arr.to_pylist())
    tm.assert_series_equal(arr.to_pandas(), expected)
    arr = arr.take([0, 1, None, 3])
    expected[2] = None
    tm.assert_series_equal(arr.to_pandas(), expected)