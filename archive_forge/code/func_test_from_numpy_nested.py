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
def test_from_numpy_nested(self):
    dt = np.dtype([('x', np.dtype([('xx', np.int8), ('yy', np.bool_)])), ('y', np.int16), ('z', np.object_)])
    assert dt.itemsize == 12
    ty = pa.struct([pa.field('x', pa.struct([pa.field('xx', pa.int8()), pa.field('yy', pa.bool_())])), pa.field('y', pa.int16()), pa.field('z', pa.string())])
    data = np.array([], dtype=dt)
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == []
    data = np.array([((1, True), 2, 'foo'), ((3, False), 4, 'bar')], dtype=dt)
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == [{'x': {'xx': 1, 'yy': True}, 'y': 2, 'z': 'foo'}, {'x': {'xx': 3, 'yy': False}, 'y': 4, 'z': 'bar'}]