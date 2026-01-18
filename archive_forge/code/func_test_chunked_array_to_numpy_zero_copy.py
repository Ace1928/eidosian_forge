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
def test_chunked_array_to_numpy_zero_copy():
    elements = [[2, 2, 4], [4, 5, 100]]
    chunked_arr = pa.chunked_array(elements)
    msg = 'zero_copy_only must be False for pyarrow.ChunkedArray.to_numpy'
    with pytest.raises(ValueError, match=msg):
        chunked_arr.to_numpy(zero_copy_only=True)
    np_arr = chunked_arr.to_numpy()
    expected = [2, 2, 4, 4, 5, 100]
    np.testing.assert_array_equal(np_arr, expected)