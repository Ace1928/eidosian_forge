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
@pytest.mark.parametrize('layout', [Card, Column, Row, Tabs, Spacer])
def test_layout_model_cache_cleanup(layout, document, comm):
    l = layout()
    model = l.get_root(document, comm)
    assert model.ref['id'] in l._models
    assert l._models[model.ref['id']] == (model, None)
    l._cleanup(model)
    assert l._models == {}