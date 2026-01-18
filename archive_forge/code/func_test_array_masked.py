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
def test_array_masked():
    arr = pa.array([4, None, 4, 3.0], mask=np.array([False, True, False, True]))
    assert arr.type == pa.int64()
    arr = pa.array(np.array([4, None, 4, 3.0], dtype='O'), mask=np.array([False, True, False, True]))
    assert arr.type == pa.int64()