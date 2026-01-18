import pytest
from bokeh.models import (
import panel as pn
from panel.layout import Tabs
def test_empty_tabs_append(document, comm):
    tabs = Tabs()
    model = tabs.get_root(document, comm=comm)
    div1 = Div()
    tabs.append(('test title', div1))
    assert len(model.tabs) == 1
    assert model.tabs[0].title == 'test title'