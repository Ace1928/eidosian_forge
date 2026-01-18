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
def test_dataframe_widget(dataframe, document, comm):
    table = DataFrame(dataframe)
    model = table.get_root(document, comm)
    index_col, int_col, float_col, str_col = model.columns
    assert index_col.title == 'index'
    assert isinstance(index_col.formatter, NumberFormatter)
    assert isinstance(index_col.editor, CellEditor)
    assert int_col.title == 'int'
    assert isinstance(int_col.formatter, NumberFormatter)
    assert isinstance(int_col.editor, IntEditor)
    assert float_col.title == 'float'
    assert isinstance(float_col.formatter, NumberFormatter)
    assert isinstance(float_col.editor, NumberEditor)
    assert str_col.title == 'str'
    assert isinstance(float_col.formatter, StringFormatter)
    assert isinstance(float_col.editor, NumberEditor)