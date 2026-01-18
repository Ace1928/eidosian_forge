from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_merge_attr_retention(self) -> None:
    da1 = create_test_dataarray_attrs(var='var1')
    da2 = create_test_dataarray_attrs(var='var2')
    da2.attrs = {'wrong': 'attributes'}
    original_attrs = da1.attrs
    result = merge([da1, da2])
    assert result.attrs == original_attrs