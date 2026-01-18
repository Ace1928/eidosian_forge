import asyncio
import os
import pandas as pd
import param
import pytest
from bokeh.models import (
from packaging.version import Version
from panel import config
from panel.depends import bind
from panel.io.state import set_curdoc, state
from panel.layout import Row, Tabs
from panel.models import HTML as BkHTML
from panel.pane import (
from panel.param import (
from panel.tests.util import mpl_available, mpl_figure
from panel.widgets import (
def test_expand_param_subobject(document, comm):

    class Test(param.Parameterized):
        a = param.Parameter()
    test = Test(a=Test(name='Nested'))
    test_pane = Param(test)
    model = test_pane.get_root(document, comm=comm)
    toggle = model.children[1].children[1]
    assert isinstance(toggle, Toggle)
    test_pane._widgets['a'][1].value = True
    assert len(model.children) == 3
    _, _, subpanel = test_pane.layout.objects
    col = model.children[2]
    assert isinstance(col, BkColumn)
    assert isinstance(col, BkColumn)
    assert len(col.children) == 2
    div, widget = col.children
    assert div.text == '<b>Nested</b>'
    assert isinstance(widget, BkTextInput)
    test_pane._widgets['a'][1].value = False
    assert len(model.children) == 2