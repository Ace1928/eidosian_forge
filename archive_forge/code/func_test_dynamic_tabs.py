import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_dynamic_tabs(document, comm, tabs):
    tabs.dynamic = True
    model = tabs.get_root(document, comm=comm)
    div1, div2 = tabs
    tab1, tab2 = model.tabs
    assert tab1.child is div1.object
    assert isinstance(tab2.child, BkSpacer)
    tabs.active = 1
    tab1, tab2 = model.tabs
    assert isinstance(tab1.child, BkSpacer)
    assert tab2.child is div2.object
    tabs.dynamic = False
    tab1, tab2 = model.tabs
    assert tab1.child is div1.object
    assert tab2.child is div2.object
    tabs.param.update(dynamic=True, active=0)
    tab1, tab2 = model.tabs
    assert tab1.child is div1.object
    assert isinstance(tab2.child, BkSpacer)