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
def test_embed_merged_sliders(document, comm):
    s1 = IntSlider(name='A', start=1, end=10, value=1)
    t1 = StaticText()
    s1.param.watch(lambda event: setattr(t1, 'value', event.new), 'value')
    s2 = IntSlider(name='A', start=1, end=10, value=1)
    t2 = StaticText()
    s2.param.watch(lambda event: setattr(t2, 'value', event.new), 'value')
    panel = Row(s1, s2, t1, t2)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    state_model = embed_state(panel, model, document)
    assert len(document.roots) == 2
    assert model is document.roots[0]
    cbs = list(model.select({'type': CustomJS}))
    assert len(cbs) == 5
    ref1, ref2 = (model.children[2].ref['id'], model.children[3].ref['id'])
    ref3 = model.children[0].children[0].ref['id']
    ref4 = model.children[1].children[0].ref['id']
    state0 = json.loads(state_model.state[0]['content'])['events']
    assert state0 == [{'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref3}, 'new': 'A: <b>1</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref1}, 'new': '1'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref4}, 'new': 'A: <b>1</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref2}, 'new': '1'}]
    state1 = json.loads(state_model.state[1]['content'])['events']
    assert state1 == [{'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref3}, 'new': 'A: <b>5</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref1}, 'new': '5'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref4}, 'new': 'A: <b>5</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref2}, 'new': '5'}]
    state2 = json.loads(state_model.state[2]['content'])['events']
    assert state2 == [{'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref3}, 'new': 'A: <b>9</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref1}, 'new': '9'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref4}, 'new': 'A: <b>9</b>'}, {'attr': 'text', 'kind': 'ModelChanged', 'model': {'id': ref2}, 'new': '9'}]