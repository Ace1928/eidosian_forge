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
def test_expand_param_subobject_tabs(document, comm):

    class Test(param.Parameterized):
        abc = param.Parameter()
    test = Test(abc=Test(name='Nested'), name='A')
    test_pane = Param(test, expand_layout=Tabs)
    model = test_pane.get_root(document, comm=comm)
    toggle = model.tabs[0].child.children[0].children[1]
    assert isinstance(toggle, Toggle)
    test_pane._widgets['abc'][1].value = True
    assert len(model.tabs) == 2
    _, subpanel = test_pane.layout.objects
    subtabs = model.tabs[1].child
    assert model.tabs[1].title == 'Abc'
    assert isinstance(subtabs, BkTabs)
    assert len(subtabs.tabs) == 1
    assert subtabs.tabs[0].title == 'Nested'
    box = subtabs.tabs[0].child
    assert isinstance(box, BkColumn)
    assert len(box.children) == 1
    widget = box.children[0]
    assert isinstance(widget, BkTextInput)
    test_pane._widgets['abc'][1].value = False
    assert len(model.tabs) == 1