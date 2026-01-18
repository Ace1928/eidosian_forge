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
def test_column_of_lists_chunked2(self):
    data1 = [[0, 1], [2, 3], [4, 5], [6, 7], [10, 11], [12, 13], [14, 15], [16, 17]]
    data2 = [[8, 9], [18, 19]]
    a1 = pa.array(data1)
    a2 = pa.array(data2)
    t1 = pa.Table.from_arrays([a1], names=['a'])
    t2 = pa.Table.from_arrays([a2], names=['a'])
    concatenated = pa.concat_tables([t1, t2])
    result = concatenated.to_pandas()
    expected = pd.DataFrame({'a': data1 + data2})
    tm.assert_frame_equal(result, expected)