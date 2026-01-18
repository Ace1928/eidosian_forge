from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_display_dataarray_style_html(self) -> None:
    da = create_test_dataarray_attrs()
    with xarray.set_options(display_style='html'):
        html = da._repr_html_()
        assert html.startswith('<div>')
        assert '#x27;nested&#x27;' in html