import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane(document, comm):
    pane = Markdown('**Markdown**')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text.endswith('&lt;p&gt;&lt;strong&gt;Markdown&lt;/strong&gt;&lt;/p&gt;\n')
    pane.object = '*Markdown*'
    assert pane._models[model.ref['id']][0] is model
    assert model.text.endswith('&lt;p&gt;&lt;em&gt;Markdown&lt;/em&gt;&lt;/p&gt;\n')
    pane._cleanup(model)
    assert pane._models == {}