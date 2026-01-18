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
def test_infer_numpy_array(self):
    data = OrderedDict([('ints', [np.array([0, 1], dtype=np.int64), np.array([2, 3], dtype=np.int64)])])
    df = pd.DataFrame(data)
    expected_schema = pa.schema([pa.field('ints', pa.list_(pa.int64()))])
    _check_pandas_roundtrip(df, expected_schema=expected_schema)