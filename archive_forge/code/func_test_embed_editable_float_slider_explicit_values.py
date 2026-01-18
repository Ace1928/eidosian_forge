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
def test_embed_editable_float_slider_explicit_values(document, comm):
    select = EditableFloatSlider()
    string = Str()

    def link(target, event):
        target.object = event.new
    select.link(string, callbacks={'value': link})
    panel = Row(select, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document, states={select: [0.1, 0.7, 1]})
    _, state = document.roots
    assert set(state.state) == {0, 1, 2}
    states = {0: 0.1, 1: 0.7, 2: 1}
    for k, v in state.state.items():
        content = json.loads(v['content'])
        assert 'events' in content
        events = content['events']
        assert len(events) == 4
        event1, event2, event3, event4 = events
        assert event1['kind'] == 'ModelChanged'
        assert event1['attr'] == 'text'
        assert event1['model'] == model.children[0].children[1].children[0].ref
        assert event1['new'] == '<b>%s</b>' % states[k]
        assert event2['kind'] == 'ModelChanged'
        assert event2['attr'] == 'value'
        assert event2['new'] == states[k]
        assert event3['kind'] == 'ModelChanged'
        assert event3['attr'] == 'value'
        assert event3['model'] == model.children[0].children[0].children[1].ref
        assert event3['new'] == states[k]
        assert event4['kind'] == 'ModelChanged'
        assert event4['attr'] == 'text'
        assert event4['model'] == model.children[1].ref
        assert event4['new'] == f'&lt;pre&gt;{states[k]}&lt;/pre&gt;'