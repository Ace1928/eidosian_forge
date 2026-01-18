from io import StringIO
from html.parser import HTMLParser
import json
import os
import re
import tempfile
import shutil
import traitlets
from ..widgets import IntSlider, IntText, Text, Widget, jslink, HBox, widget_serialization, widget as widget_module
from ..embed import embed_data, embed_snippet, embed_minimal_html, dependency_state
def test_embed_data_simple(self):
    w = IntText(4)
    state = dependency_state(w, drop_defaults=True)
    data = embed_data(views=w, drop_defaults=True, state=state)
    state = data['manager_state']['state']
    views = data['view_specs']
    assert len(state) == 3
    assert len(views) == 1
    model_names = [s['model_name'] for s in state.values()]
    assert 'IntTextModel' in model_names