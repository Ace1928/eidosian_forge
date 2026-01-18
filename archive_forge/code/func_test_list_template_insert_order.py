import json
import param
import pytest
from bokeh.document import Document
from bokeh.io.doc import patch_curdoc
from panel.layout import GridSpec, Row
from panel.pane import HoloViews, Markdown
from panel.template import (
from panel.template.base import BasicTemplate
from panel.widgets import FloatSlider
from .util import hv_available
@pytest.mark.parametrize(['template_class'], [(t,) for t in LIST_TEMPLATES])
def test_list_template_insert_order(template_class):
    template = template_class()
    template.main.append(1)
    template.main.insert(0, 0)
    template.main.extend([2, 3])
    objs = list(template._render_items.values())[4:]
    (obj1, tag1), (obj2, tag2), (obj3, tag3), (obj4, tag4) = objs
    assert tag1 == tag2 == tag3 == tag4 == ['main']
    assert obj1.object == 0
    assert obj2.object == 1
    assert obj3.object == 2
    assert obj4.object == 3