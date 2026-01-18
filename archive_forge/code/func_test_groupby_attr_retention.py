from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_groupby_attr_retention(self) -> None:
    da = xarray.DataArray([1, 2, 3], [('x', [1, 1, 2])])
    da.attrs = {'attr1': 5, 'attr2': 'history', 'attr3': {'nested': 'more_info'}}
    original_attrs = da.attrs
    result = da.groupby('x').sum(keep_attrs=True)
    assert result.attrs == original_attrs
    with xarray.set_options(keep_attrs='default'):
        result = da.groupby('x').sum(keep_attrs=True)
        assert result.attrs == original_attrs
    with xarray.set_options(keep_attrs=True):
        result1 = da.groupby('x')
        result = result1.sum()
        assert result.attrs == original_attrs
    with xarray.set_options(keep_attrs=False):
        result = da.groupby('x').sum()
        assert result.attrs == {}