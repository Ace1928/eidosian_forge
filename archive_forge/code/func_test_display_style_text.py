from __future__ import annotations
import pytest
import xarray
from xarray import concat, merge
from xarray.backends.file_manager import FILE_CACHE
from xarray.core.options import OPTIONS, _get_keep_attrs
from xarray.tests.test_dataset import create_test_data
def test_display_style_text(self) -> None:
    ds = create_test_dataset_attrs()
    with xarray.set_options(display_style='text'):
        text = ds._repr_html_()
        assert text.startswith('<pre>')
        assert '&#x27;nested&#x27;' in text
        assert '&lt;xarray.Dataset&gt;' in text