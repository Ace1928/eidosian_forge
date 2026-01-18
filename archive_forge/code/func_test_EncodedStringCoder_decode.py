from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_EncodedStringCoder_decode() -> None:
    coder = strings.EncodedStringCoder()
    raw_data = np.array([b'abc', 'ß∂µ∆'.encode()])
    raw = Variable(('x',), raw_data, {'_Encoding': 'utf-8'})
    actual = coder.decode(raw)
    expected = Variable(('x',), np.array(['abc', 'ß∂µ∆'], dtype=object))
    assert_identical(actual, expected)
    assert_identical(coder.decode(actual[0]), expected[0])