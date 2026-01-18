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
def test_concat_array_invalid_type():
    with pytest.raises(TypeError, match='should contain Array objects'):
        pa.concat_arrays([None])
    arr = pa.chunked_array([[0, 1], [3, 4]])
    with pytest.raises(TypeError, match='should contain Array objects'):
        pa.concat_arrays(arr)