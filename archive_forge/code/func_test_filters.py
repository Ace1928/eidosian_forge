from __future__ import annotations
import os.path
import pytest
from dask.utils import format_bytes
from dask.widgets import FILTERS, TEMPLATE_PATHS, get_environment, get_template
def test_filters():
    template = get_template('bytes.html.j2')
    assert format_bytes in FILTERS.values()
    assert format_bytes(2000000000.0) in template.render(foo=2000000000.0)
    template = get_template('custom_filter.html.j2')
    assert 'baz' in template.render(foo=None)