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
def test_array_from_invalid_dim_raises():
    msg = 'only handle 1-dimensional arrays'
    arr2d = np.array([[1, 2, 3], [4, 5, 6]])
    with pytest.raises(ValueError, match=msg):
        pa.array(arr2d)
    arr0d = np.array(0)
    with pytest.raises(ValueError, match=msg):
        pa.array(arr0d)