from __future__ import annotations
import datetime as dt
from contextlib import contextmanager
import numpy as np
import pandas as pd
import param
import pytest
from bokeh.models.widgets.tables import (
from playwright.sync_api import expect
from panel.depends import bind
from panel.io.state import state
from panel.layout.base import Column
from panel.models.tabulator import _TABULATOR_THEMES_MAPPING
from panel.tests.util import get_ctrl_modifier, serve_component, wait_until
from panel.widgets import Select, Tabulator
def count_per_page(count: int, page_size: int):
    """
    >>> count_per_page(12, 7)
    [7, 5]
    """
    original_count = count
    count_per_page = []
    while True:
        page_count = min(count, page_size)
        count_per_page.append(page_count)
        count -= page_count
        if count == 0:
            break
    assert sum(count_per_page) == original_count
    return count_per_page