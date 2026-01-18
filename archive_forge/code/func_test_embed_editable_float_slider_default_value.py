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
def test_embed_editable_float_slider_default_value(document, comm):
    slider = EditableFloatSlider(start=0, end=7.2, value=3.6)
    string = Str()

    def link(target, event):
        target.object = event.new
    slider.link(string, callbacks={'value': link})
    panel = Row(slider, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    embed_state(panel, model, document)
    layout, state = document.roots
    assert set(state.state) == {0, 1, 2}
    assert layout.children[0].children[1].children[1].value == 1