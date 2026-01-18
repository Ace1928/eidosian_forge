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
def test_nested_lists_all_none(self):
    data = np.array([[None, None], None], dtype=object)
    arr = pa.array(data)
    expected = pa.array(list(data))
    assert arr.equals(expected)
    assert arr.type == pa.list_(pa.null())
    data2 = np.array([None, None, [None, None], np.array([None, None], dtype=object)], dtype=object)
    arr = pa.array(data2)
    expected = pa.array([None, None, [None, None], [None, None]])
    assert arr.equals(expected)