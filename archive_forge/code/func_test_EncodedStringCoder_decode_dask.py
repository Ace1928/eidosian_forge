from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@requires_dask
def test_EncodedStringCoder_decode_dask() -> None:
    coder = strings.EncodedStringCoder()
    raw_data = np.array([b'abc', 'ß∂µ∆'.encode()])
    raw = Variable(('x',), raw_data, {'_Encoding': 'utf-8'}).chunk()
    actual = coder.decode(raw)
    assert isinstance(actual.data, da.Array)
    expected = Variable(('x',), np.array(['abc', 'ß∂µ∆'], dtype=object))
    assert_identical(actual, expected)
    actual_indexed = coder.decode(actual[0])
    assert isinstance(actual_indexed.data, da.Array)
    assert_identical(actual_indexed, expected[0])