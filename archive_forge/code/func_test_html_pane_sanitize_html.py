import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_html_pane_sanitize_html(document, comm):
    pane = HTML('<h1><strong>HTML</h1></strong>', sanitize_html=True)
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text.endswith('&lt;strong&gt;HTML&lt;/strong&gt;')
    pane.sanitize_html = False
    assert model.text.endswith('&lt;h1&gt;&lt;strong&gt;HTML&lt;/h1&gt;&lt;/strong&gt;')