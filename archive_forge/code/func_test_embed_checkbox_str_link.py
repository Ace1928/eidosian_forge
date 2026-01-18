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
def test_embed_checkbox_str_link(document, comm):
    checkbox = Checkbox()
    string = Str()

    def link(target, event):
        target.object = event.new
    checkbox.link(string, callbacks={'value': link})
    panel = Row(checkbox, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    _, state = document.roots
    assert set(state.state) == {False, True}
    for k, v in state.state.items():
        content = json.loads(v['content'])
        assert 'events' in content
        events = content['events']
        assert len(events) == 1
        event = events[0]
        assert event['kind'] == 'ModelChanged'
        assert event['attr'] == 'text'
        assert event['model'] == model.children[1].ref
        assert event['new'] == f'&lt;pre&gt;{k}&lt;/pre&gt;'