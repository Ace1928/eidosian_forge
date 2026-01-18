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
def test_infer_lists(self):
    data = OrderedDict([('nan_ints', [[np.nan, 1], [2, 3]]), ('ints', [[0, 1], [2, 3]]), ('strs', [[None, 'b'], ['c', 'd']]), ('nested_strs', [[[None, 'b'], ['c', 'd']], None])])
    df = pd.DataFrame(data)
    expected_schema = pa.schema([pa.field('nan_ints', pa.list_(pa.int64())), pa.field('ints', pa.list_(pa.int64())), pa.field('strs', pa.list_(pa.string())), pa.field('nested_strs', pa.list_(pa.list_(pa.string())))])
    _check_pandas_roundtrip(df, expected_schema=expected_schema)