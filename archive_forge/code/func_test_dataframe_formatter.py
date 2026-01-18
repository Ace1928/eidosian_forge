import asyncio
import datetime as dt
import numpy as np
import pandas as pd
import pytest
from bokeh.models.widgets.tables import (
from packaging.version import Version
from panel.depends import bind
from panel.io.state import set_curdoc
from panel.models.tabulator import CellClickEvent, TableEditEvent
from panel.tests.util import mpl_available, serve_and_request, wait_until
from panel.util import BOKEH_JS_NAT
from panel.widgets import Button, TextInput
from panel.widgets.tables import DataFrame, Tabulator
def test_dataframe_formatter(dataframe, document, comm):
    formatter = NumberFormatter(format='0.0000')
    table = DataFrame(dataframe, formatters={'float': formatter})
    model = table.get_root(document, comm)
    model_formatter = model.columns[2].formatter
    assert model_formatter is not formatter
    assert isinstance(model_formatter, NumberFormatter)
    assert model_formatter.format == formatter.format