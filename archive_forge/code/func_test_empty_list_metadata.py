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
def test_empty_list_metadata(self):
    c1 = [['test'], ['a', 'b'], None]
    c2 = [[], [], []]
    arrays = OrderedDict([('c1', pa.array(c1, type=pa.list_(pa.string()))), ('c2', pa.array(c2, type=pa.list_(pa.string())))])
    rb = pa.RecordBatch.from_arrays(list(arrays.values()), list(arrays.keys()))
    tbl = pa.Table.from_batches([rb])
    df = tbl.to_pandas()
    tbl2 = pa.Table.from_pandas(df)
    md2 = tbl2.schema.pandas_metadata
    df2 = tbl2.to_pandas()
    expected = pd.DataFrame(OrderedDict([('c1', c1), ('c2', c2)]))
    tm.assert_frame_equal(df2, expected)
    assert md2['columns'] == [{'name': 'c1', 'field_name': 'c1', 'metadata': None, 'numpy_type': 'object', 'pandas_type': 'list[unicode]'}, {'name': 'c2', 'field_name': 'c2', 'metadata': None, 'numpy_type': 'object', 'pandas_type': 'list[empty]'}]