from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_concat_attr_retention(self) -> None:
    ds1 = create_test_dataset_attrs()
    ds2 = create_test_dataset_attrs()
    ds2.attrs = {'wrong': 'attributes'}
    original_attrs = ds1.attrs
    result = concat([ds1, ds2], dim='dim1')
    assert result.attrs == original_attrs