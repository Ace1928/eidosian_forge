from __future__ import annotations
import os.path
import pytest
from dask.utils import format_bytes
from dask.widgets import FILTERS, TEMPLATE_PATHS, get_environment, get_template
def test_environment():
    environment = get_environment()
    assert isinstance(environment, jinja2.Environment)