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
@pytest.mark.parametrize('text_align', [{'A': 'center'}, 'center'], ids=['dict', 'str'])
def test_bokeh_formatter_column_with_no_textalign_but_text_align_set(document, comm, text_align):
    df = pd.DataFrame({'A': [1, 2, 3]})
    table = Tabulator(df, formatters=dict(A=HTMLTemplateFormatter(template='<b><%= value %>"></b>')), text_align=text_align)
    model = table.get_root(document, comm)
    assert model.configuration['columns'][1]['hozAlign'] == 'center'