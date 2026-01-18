import glob
import json
import os
from io import StringIO
import pytest
from bokeh.models import CustomJS
from panel import Row
from panel.config import config
from panel.io.embed import embed_state
from panel.pane import Str
from panel.param import Param
from panel.widgets import (
def test_embed_slider_str_link(document, comm):
    slider = FloatSlider(start=0, end=10)
    string = Str()

    def link(target, event):
        target.object = event.new
    slider.link(string, callbacks={'value': link})
    panel = Row(slider, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    _, state = document.roots
    assert set(state.state) == {0, 1, 2}
    values = [0, 5, 10]
    for k, v in state.state.items():
        content = json.loads(v['content'])
        assert 'events' in content
        events = content['events']
        assert len(events) == 2
        event1, event2 = events
        assert event1['kind'] == 'ModelChanged'
        assert event1['attr'] == 'text'
        assert event1['model'] == model.children[0].children[0].ref
        assert event1['new'] == '<b>%d</b>' % values[k]
        assert event2['kind'] == 'ModelChanged'
        assert event2['attr'] == 'text'
        assert event2['model'] == model.children[1].ref
        assert event2['new'] == '&lt;pre&gt;%.1f&lt;/pre&gt;' % values[k]