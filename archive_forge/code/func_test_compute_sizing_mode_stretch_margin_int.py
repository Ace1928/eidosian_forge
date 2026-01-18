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
@pytest.mark.parametrize('dim', ['width', 'height'])
def test_compute_sizing_mode_stretch_margin_int(dim, document, comm):
    margin = 10
    md = Markdown(**{dim: 100})
    col = Column(md, margin=margin, sizing_mode=f'stretch_{dim}')
    root = col.get_root(document, comm=comm)
    new_props = col._compute_sizing_mode(root.children, {'margin': margin})
    assert new_props == {f'min_{dim}': 120, 'sizing_mode': f'stretch_{dim}'}