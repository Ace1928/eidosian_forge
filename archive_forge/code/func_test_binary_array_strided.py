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
def test_binary_array_strided():
    nparray = np.array([b'ab', b'cd', b'ef'])
    arrow_array = pa.array(nparray[::2], pa.binary(2), mask=np.array([False, False]))
    assert [b'ab', b'ef'] == arrow_array.to_pylist()
    nparray = np.array([b'ab', b'cd', b'ef'])
    arrow_array = pa.array(nparray[::2], pa.binary(2))
    assert [b'ab', b'ef'] == arrow_array.to_pylist()