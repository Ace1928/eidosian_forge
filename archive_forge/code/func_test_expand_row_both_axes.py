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
def test_expand_row_both_axes(document, comm):
    div1 = Div(sizing_mode='stretch_both')
    div2 = Div(sizing_mode='stretch_both')
    layout = Row(div1, div2)
    model = layout.get_root(document, comm)
    assert model.sizing_mode == 'stretch_both'