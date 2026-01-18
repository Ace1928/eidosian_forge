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
def test_minimal_html_filehandle(self):
    w = IntText(4)
    output = StringIO()
    state = dependency_state(w, drop_defaults=True)
    embed_minimal_html(output, views=w, drop_defaults=True, state=state)
    content = output.getvalue()
    assert content.splitlines()[0] == '<!DOCTYPE html>'