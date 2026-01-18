from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_display_style() -> None:
    original = 'html'
    assert OPTIONS['display_style'] == original
    with pytest.raises(ValueError):
        xarray.set_options(display_style='invalid_str')
    with xarray.set_options(display_style='text'):
        assert OPTIONS['display_style'] == 'text'
    assert OPTIONS['display_style'] == original