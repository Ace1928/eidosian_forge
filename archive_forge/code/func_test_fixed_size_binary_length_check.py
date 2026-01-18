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
def test_fixed_size_binary_length_check():
    data = [b'\x19h\r\x9e\x00\x00\x00\x00\x01\x9b\x9fA']
    assert len(data[0]) == 12
    ty = pa.binary(12)
    arr = pa.array(data, type=ty)
    assert arr.to_pylist() == data