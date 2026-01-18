from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
def test_cast_date32_to_int():
    arr = pa.array([0, 1, 2], type='i4')
    result1 = arr.cast('date32')
    result2 = result1.cast('i4')
    expected1 = pa.array([datetime.date(1970, 1, 1), datetime.date(1970, 1, 2), datetime.date(1970, 1, 3)]).cast('date32')
    assert result1.equals(expected1)
    assert result2.equals(arr)