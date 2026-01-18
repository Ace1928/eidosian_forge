from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
def test_EncodedStringCoder_encode() -> None:
    dtype = strings.create_vlen_dtype(str)
    raw_data = np.array(['abc', 'ß∂µ∆'], dtype=dtype)
    expected_data = np.array([r.encode('utf-8') for r in raw_data], dtype=object)
    coder = strings.EncodedStringCoder(allows_unicode=True)
    raw = Variable(('x',), raw_data, encoding={'dtype': 'S1'})
    actual = coder.encode(raw)
    expected = Variable(('x',), expected_data, attrs={'_Encoding': 'utf-8'})
    assert_identical(actual, expected)
    raw = Variable(('x',), raw_data)
    assert_identical(coder.encode(raw), raw)
    coder = strings.EncodedStringCoder(allows_unicode=False)
    assert_identical(coder.encode(raw), expected)