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
@pytest.mark.parametrize('panel', [Card, Column, Row, Tabs, Accordion])
def test_expand_only_non_fixed_axis_height(panel, document, comm):
    div1 = Div(sizing_mode='stretch_width')
    div2 = Div(sizing_mode='stretch_height')
    layout = panel(div1, div2, height=500)
    model = layout.get_root(document, comm)
    assert model.sizing_mode == 'stretch_width'