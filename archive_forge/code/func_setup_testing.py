from __future__ import annotations
import os.path
import pytest
from dask.utils import format_bytes
from dask.widgets import FILTERS, TEMPLATE_PATHS, get_environment, get_template
@pytest.fixture(autouse=True)
def setup_testing():
    TEMPLATE_PATHS.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))
    FILTERS['custom_filter'] = lambda x: 'baz'