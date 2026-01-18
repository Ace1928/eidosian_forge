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
def test_embed_select_str_explicit_values_not_found(document, comm):
    select = Select(options=['A', 'B', 'C'])
    string = Str()

    def link(target, event):
        target.object = event.new
    select.link(string, callbacks={'value': link})
    panel = Row(select, string)
    with config.set(embed=True):
        model = panel.get_root(document, comm)
    with pytest.raises(ValueError):
        embed_state(panel, model, document, states={select: ['A', 'D']})