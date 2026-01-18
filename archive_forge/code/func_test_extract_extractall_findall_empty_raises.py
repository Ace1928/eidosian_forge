from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_extractall_findall_empty_raises(dtype) -> None:
    pat_str = dtype('.*')
    pat_re = re.compile(pat_str)
    value = xr.DataArray([['a']], dims=['X', 'Y']).astype(dtype)
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.extract(pat=pat_str, dim='ZZ')
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.extract(pat=pat_re, dim='ZZ')
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.extractall(pat=pat_str, group_dim='XX', match_dim='YY')
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.extractall(pat=pat_re, group_dim='XX', match_dim='YY')
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.findall(pat=pat_str)
    with pytest.raises(ValueError, match='No capture groups found in pattern.'):
        value.str.findall(pat=pat_re)