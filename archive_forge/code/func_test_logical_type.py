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
@pytest.mark.parametrize(('type', 'expected'), [(pa.null(), 'empty'), (pa.bool_(), 'bool'), (pa.int8(), 'int8'), (pa.int16(), 'int16'), (pa.int32(), 'int32'), (pa.int64(), 'int64'), (pa.uint8(), 'uint8'), (pa.uint16(), 'uint16'), (pa.uint32(), 'uint32'), (pa.uint64(), 'uint64'), (pa.float16(), 'float16'), (pa.float32(), 'float32'), (pa.float64(), 'float64'), (pa.date32(), 'date'), (pa.date64(), 'date'), (pa.binary(), 'bytes'), (pa.binary(length=4), 'bytes'), (pa.string(), 'unicode'), (pa.list_(pa.list_(pa.int16())), 'list[list[int16]]'), (pa.decimal128(18, 3), 'decimal'), (pa.timestamp('ms'), 'datetime'), (pa.timestamp('us', 'UTC'), 'datetimetz'), (pa.time32('s'), 'time'), (pa.time64('us'), 'time')])
def test_logical_type(type, expected):
    assert get_logical_type(type) == expected