from __future__ import annotations
import re
from typing import Callable
import numpy as np
import pytest
import xarray as xr
from xarray.tests import assert_equal, assert_identical, requires_dask
def test_extract_extractall_findall_case_re_raises(dtype) -> None:
    pat_str = '.*'
    pat_re = re.compile(pat_str)
    value = xr.DataArray([['a']], dims=['X', 'Y']).astype(dtype)
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.extract(pat=pat_re, case=True, dim='ZZ')
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.extract(pat=pat_re, case=False, dim='ZZ')
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.extractall(pat=pat_re, case=True, group_dim='XX', match_dim='YY')
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.extractall(pat=pat_re, case=False, group_dim='XX', match_dim='YY')
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.findall(pat=pat_re, case=True)
    with pytest.raises(ValueError, match='Case cannot be set when pat is a compiled regex.'):
        value.str.findall(pat=pat_re, case=False)