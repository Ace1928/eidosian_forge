from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_dataset_attr_retention(self) -> None:
    ds = create_test_dataset_attrs()
    original_attrs = ds.attrs
    result = ds.mean()
    assert result.attrs == {}
    with xarray.set_options(keep_attrs='default'):
        result = ds.mean()
        assert result.attrs == {}
    with xarray.set_options(keep_attrs=True):
        result = ds.mean()
        assert result.attrs == original_attrs
    with xarray.set_options(keep_attrs=False):
        result = ds.mean()
        assert result.attrs == {}