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
def test_metadata_compat_missing_field_name():
    a_values = [1, 2, 3, 4]
    b_values = ['a', 'b', 'c', 'd']
    a_arrow = pa.array(a_values, type='int64')
    b_arrow = pa.array(b_values, type='utf8')
    expected = pd.DataFrame({'a': a_values, 'b': b_values}, index=pd.RangeIndex(0, 8, step=2, name='qux'))
    table = pa.table({'a': a_arrow, 'b': b_arrow})
    table = table.replace_schema_metadata({b'pandas': json.dumps({'column_indexes': [{'field_name': None, 'metadata': None, 'name': None, 'numpy_type': 'object', 'pandas_type': 'mixed-integer'}], 'columns': [{'metadata': None, 'name': 'a', 'numpy_type': 'int64', 'pandas_type': 'int64'}, {'metadata': None, 'name': 'b', 'numpy_type': 'object', 'pandas_type': 'unicode'}], 'index_columns': [{'kind': 'range', 'name': 'qux', 'start': 0, 'step': 2, 'stop': 8}], 'pandas_version': '0.25.0'})})
    result = table.to_pandas()
    tm.assert_frame_equal(result, expected)