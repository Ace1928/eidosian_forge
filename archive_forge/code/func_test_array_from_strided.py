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
def test_array_from_strided():
    pydata = [([b'ab', b'cd', b'ef'], (pa.binary(), pa.binary(2))), ([1, 2, 3], (pa.int8(), pa.int16(), pa.int32(), pa.int64())), ([1.0, 2.0, 3.0], (pa.float32(), pa.float64())), (['ab', 'cd', 'ef'], (pa.utf8(),))]
    for values, dtypes in pydata:
        nparray = np.array(values)
        for patype in dtypes:
            for mask in (None, np.array([False, False])):
                arrow_array = pa.array(nparray[::2], patype, mask=mask)
                assert values[::2] == arrow_array.to_pylist()