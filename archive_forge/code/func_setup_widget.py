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
@pytest.fixture(autouse=True)
def setup_widget(self, page):
    self.widget = Tabulator(value=pd.DataFrame(np.arange(20) + 100), disabled=True, pagination='remote', page_size=10, selectable=self.selectable, header_filters=True)
    serve_component(page, self.widget)