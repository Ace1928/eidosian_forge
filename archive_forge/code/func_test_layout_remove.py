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
def test_layout_remove(panel, document, comm):
    div1 = Div()
    div2 = Div()
    layout = panel(div1, div2)
    p1, p2 = layout.objects
    model = layout.get_root(document, comm=comm)
    assert p1._models[model.ref['id']][0] is model.children[0]
    layout.remove(p1)
    assert model.children == [div2]
    assert p1._models == {}