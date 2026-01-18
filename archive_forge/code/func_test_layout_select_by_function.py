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
@pytest.mark.parametrize('panel', [Card, Column, Row])
def test_layout_select_by_function(panel):
    div1 = Div()
    div2 = Div()
    layout = panel(div1, div2)
    panes = layout.select(lambda x: getattr(x, 'object', None) is div2)
    assert len(panes) == 1
    assert panes[0].object is div2