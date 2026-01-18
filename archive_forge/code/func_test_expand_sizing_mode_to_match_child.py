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
@pytest.mark.parametrize('panel', [Card, Column, Tabs, Accordion])
@pytest.mark.parametrize('sizing_mode', ['stretch_width', 'stretch_height', 'stretch_both'])
def test_expand_sizing_mode_to_match_child(panel, sizing_mode, document, comm):
    div1 = Div()
    div2 = Div(sizing_mode=sizing_mode)
    layout = panel(div1, div2)
    model = layout.get_root(document, comm)
    assert model.sizing_mode == sizing_mode