import param
import pytest
from bokeh.models import Column as BkColumn, Div, Row as BkRow
from panel.chat import ChatInterface
from panel.layout import (
from panel.layout.base import ListPanel, NamedListPanel
from panel.pane import Bokeh, Markdown
from panel.param import Param
from panel.tests.util import check_layoutable_properties
from panel.widgets import Debugger, MultiSelect
@pytest.mark.parametrize('panel', [Column, Row])
def test_layout_add_list(panel, document, comm):
    div1 = Div()
    div2 = Div()
    layout1 = panel(div1, div2)
    div3 = Div()
    div4 = Div()
    combined = layout1 + [div3, div4]
    model = combined.get_root(document, comm=comm)
    assert model.children == [div1, div2, div3, div4]