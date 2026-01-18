from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_extractall_name_collision_raises(dtype) -> None:
    pat_str = '(\\w+)'
    pat_re = re.compile(pat_str)
    value = xr.DataArray([['a']], dims=['X', 'Y']).astype(dtype)
    with pytest.raises(KeyError, match="Dimension 'X' already present in DataArray."):
        value.str.extract(pat=pat_str, dim='X')
    with pytest.raises(KeyError, match="Dimension 'X' already present in DataArray."):
        value.str.extract(pat=pat_re, dim='X')
    with pytest.raises(KeyError, match="Group dimension 'X' already present in DataArray."):
        value.str.extractall(pat=pat_str, group_dim='X', match_dim='ZZ')
    with pytest.raises(KeyError, match="Group dimension 'X' already present in DataArray."):
        value.str.extractall(pat=pat_re, group_dim='X', match_dim='YY')
    with pytest.raises(KeyError, match="Match dimension 'Y' already present in DataArray."):
        value.str.extractall(pat=pat_str, group_dim='XX', match_dim='Y')
    with pytest.raises(KeyError, match="Match dimension 'Y' already present in DataArray."):
        value.str.extractall(pat=pat_re, group_dim='XX', match_dim='Y')
    with pytest.raises(KeyError, match="Group dimension 'ZZ' is the same as match dimension 'ZZ'."):
        value.str.extractall(pat=pat_str, group_dim='ZZ', match_dim='ZZ')
    with pytest.raises(KeyError, match="Group dimension 'ZZ' is the same as match dimension 'ZZ'."):
        value.str.extractall(pat=pat_re, group_dim='ZZ', match_dim='ZZ')