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
def test_array_from_shrunken_masked():
    ma = np.ma.array([0], dtype='int64')
    result = pa.array(ma)
    expected = pa.array([0], type='int64')
    assert expected.equals(result)