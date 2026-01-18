from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_keep_attrs() -> None:
    with pytest.raises(ValueError):
        xarray.set_options(keep_attrs='invalid_str')
    with xarray.set_options(keep_attrs=True):
        assert OPTIONS['keep_attrs']
    with xarray.set_options(keep_attrs=False):
        assert not OPTIONS['keep_attrs']
    with xarray.set_options(keep_attrs='default'):
        assert _get_keep_attrs(default=True)
        assert not _get_keep_attrs(default=False)