import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.slow
@pytest.mark.large_memory
@pytest.mark.parametrize('ty', [pa.large_binary(), pa.large_string()])
def test_large_binary_value(ty):
    s = b'0123456789abcdefghijklmnopqrstuvwxyz'
    nrepeats = math.ceil((2 ** 32 + 5) / len(s))
    arr = pa.array([b'foo', s * nrepeats, None, b'bar'], type=ty)
    assert isinstance(arr, pa.Array)
    assert arr.type == ty
    assert len(arr) == 4
    buf = arr[1].as_buffer()
    assert len(buf) == len(s) * nrepeats