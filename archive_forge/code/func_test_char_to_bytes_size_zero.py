from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_char_to_bytes_size_zero() -> None:
    array = np.zeros((3, 0), dtype='S1')
    expected = np.array([b'', b'', b''])
    actual = strings.char_to_bytes(array)
    assert_array_equal(actual, expected)