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
@pytest.mark.parametrize('unit', ['ns', 'us', 'ms', 's'])
def test_array_from_list_of_timestamps(unit):
    n = np.datetime64('NaT', unit)
    x = np.datetime64('2017-01-01 01:01:01.111111111', unit)
    y = np.datetime64('2018-11-22 12:24:48.111111111', unit)
    a1 = pa.array([n, x, y])
    a2 = pa.array([n, x, y], type=pa.timestamp(unit))
    assert a1.type == a2.type
    assert a1.type.unit == unit
    assert a1[0] == a2[0]