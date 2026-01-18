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
def test_map_array_with_nulls(self):
    data = [[(b'a', 1), (b'b', 2)], None, [(b'd', 4), (b'e', 5), (b'f', None)], [(b'g', 7)]]
    expected = [[(k, float(v) if v is not None else None) for k, v in row] if row is not None else None for row in data]
    expected = pd.Series(expected)
    arr = pa.array(data, type=pa.map_(pa.binary(), pa.int32()))
    actual = arr.to_pandas()
    tm.assert_series_equal(actual, expected, check_names=False)