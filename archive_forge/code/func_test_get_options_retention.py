from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
@pytest.mark.parametrize('set_value', ['left', 'exact'])
def test_get_options_retention(set_value):
    """Test to check if get_options will return changes made by set_options"""
    with xarray.set_options(arithmetic_join=set_value):
        get_options = xarray.get_options()
        assert get_options['arithmetic_join'] == set_value