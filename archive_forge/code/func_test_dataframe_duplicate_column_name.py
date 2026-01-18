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
def test_dataframe_duplicate_column_name(document, comm):
    df = pd.DataFrame([[1, 1], [2, 2]], columns=['col', 'col'])
    with pytest.raises(ValueError):
        table = DataFrame(df)
    df = pd.DataFrame([[1, 1], [2, 2]], columns=['a', 'b'])
    table = DataFrame(df)
    with pytest.raises(ValueError):
        table.value = table.value.rename(columns={'a': 'b'})
    df = pd.DataFrame([[1, 1], [2, 2]], columns=['a', 'b'])
    table = DataFrame(df)
    table.get_root(document, comm)
    with pytest.raises(ValueError):
        table.value = table.value.rename(columns={'a': 'b'})