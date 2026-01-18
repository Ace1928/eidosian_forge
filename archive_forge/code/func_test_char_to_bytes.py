from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_char_to_bytes() -> None:
    array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']])
    expected = np.array([b'abc', b'def'])
    actual = strings.char_to_bytes(array)
    assert_array_equal(actual, expected)
    expected = np.array([b'ad', b'be', b'cf'])
    actual = strings.char_to_bytes(array.T)
    assert_array_equal(actual, expected)