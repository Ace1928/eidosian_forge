from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@pytest.mark.parametrize('data', [np.array([b'a', b'bc']), np.array([b'a', b'bc'], dtype=strings.create_vlen_dtype(bytes))])
def test_CharacterArrayCoder_encode(data) -> None:
    coder = strings.CharacterArrayCoder()
    raw = Variable(('x',), data)
    actual = coder.encode(raw)
    expected = Variable(('x', 'string2'), np.array([[b'a', b''], [b'b', b'c']]))
    assert_identical(actual, expected)