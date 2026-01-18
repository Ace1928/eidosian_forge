import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane_newline(document, comm):
    pane = Markdown("Hello\nWorld\nI'm here!", renderer='markdown-it')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;p&gt;Hello&lt;br /&gt;\nWorld&lt;br /&gt;\nI&#x27;m here!&lt;/p&gt;\n'
    pane = Markdown('Hello\n\nWorld')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;p&gt;Hello&lt;/p&gt;\n&lt;p&gt;World&lt;/p&gt;\n'
    pane = Markdown("Hello\nWorld\nI'm here!", renderer='markdown-it', renderer_options={'breaks': False})
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text == '&lt;p&gt;Hello\nWorld\nI&#x27;m here!&lt;/p&gt;\n'