from __future__ import annotations
from contextlib import suppress
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from xarray.coding import variables
from xarray.conventions import decode_cf_variable, encode_cf_variable
from xarray.tests import assert_allclose, assert_equal, assert_identical, requires_dask
@pytest.mark.parametrize('bits', [1, 2, 4, 8])
def test_decode_unsigned_from_signed(bits) -> None:
    unsigned_dtype = np.dtype(f'u{bits}')
    signed_dtype = np.dtype(f'i{bits}')
    original_values = np.array([np.iinfo(unsigned_dtype).max], dtype=unsigned_dtype)
    encoded = xr.Variable(('x',), original_values.astype(signed_dtype), attrs={'_Unsigned': 'true'})
    coder = variables.UnsignedIntegerCoder()
    decoded = coder.decode(encoded)
    assert decoded.dtype == unsigned_dtype
    assert decoded.values == original_values