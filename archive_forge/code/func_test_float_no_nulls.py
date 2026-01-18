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
def test_float_no_nulls(self):
    data = {}
    fields = []
    dtypes = [('f2', pa.float16()), ('f4', pa.float32()), ('f8', pa.float64())]
    num_values = 100
    for numpy_dtype, arrow_dtype in dtypes:
        values = np.random.randn(num_values)
        data[numpy_dtype] = values.astype(numpy_dtype)
        fields.append(pa.field(numpy_dtype, arrow_dtype))
    df = pd.DataFrame(data)
    schema = pa.schema(fields)
    _check_pandas_roundtrip(df, expected_schema=schema)