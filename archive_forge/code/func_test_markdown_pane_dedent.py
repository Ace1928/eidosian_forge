import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane_dedent(document, comm):
    pane = Markdown('    ABC')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text.endswith('&lt;p&gt;ABC&lt;/p&gt;\n')
    pane.dedent = False
    assert model.text.startswith('&lt;pre&gt;&lt;code&gt;ABC')