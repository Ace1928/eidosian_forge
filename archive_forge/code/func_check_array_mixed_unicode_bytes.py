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
def check_array_mixed_unicode_bytes(binary_type, string_type):
    values = ['qux', b'foo', bytearray(b'barz')]
    b_values = [b'qux', b'foo', b'barz']
    u_values = ['qux', 'foo', 'barz']
    arr = pa.array(values)
    expected = pa.array(b_values, type=pa.binary())
    assert arr.type == pa.binary()
    assert arr.equals(expected)
    arr = pa.array(values, type=binary_type)
    expected = pa.array(b_values, type=binary_type)
    assert arr.type == binary_type
    assert arr.equals(expected)
    arr = pa.array(values, type=string_type)
    expected = pa.array(u_values, type=string_type)
    assert arr.type == string_type
    assert arr.equals(expected)