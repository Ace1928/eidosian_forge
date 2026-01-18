from __future__ import annotations
from contextlib import suppress
import numpy as np
import pytest
from xarray import Variable
from xarray.coding import strings
from xarray.core import indexing
from xarray.tests import (
@requires_dask
def test_char_to_bytes_dask() -> None:
    numpy_array = np.array([[b'a', b'b', b'c'], [b'd', b'e', b'f']])
    array = da.from_array(numpy_array, ((2,), (3,)))
    expected = np.array([b'abc', b'def'])
    actual = strings.char_to_bytes(array)
    assert isinstance(actual, da.Array)
    assert actual.chunks == ((2,),)
    assert actual.dtype == 'S3'
    assert_array_equal(np.array(actual), expected)
    with pytest.raises(ValueError, match='stacked dask character array'):
        strings.char_to_bytes(array.rechunk(1))