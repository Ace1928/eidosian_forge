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
def test_boolean_objects_to_int(self):
    s = pd.Series([True, True, False, True, True] * 2, dtype=object)
    expected = pd.Series([1, 1, 0, 1, 1] * 2)
    expected_msg = 'Expected integer, got bool'
    with pytest.raises(pa.ArrowTypeError, match=expected_msg):
        _check_array_roundtrip(s, expected=expected, type=pa.int64())