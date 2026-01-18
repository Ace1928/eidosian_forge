import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_markdown_pane_extensions(document, comm):
    pane = Markdown('\n    ```python\n    None\n    ```\n    ', renderer='markdown')
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert 'codehilite' in model.text
    pane.extensions = ['extra', 'smarty']
    assert model.text.startswith('&lt;pre&gt;&lt;code class=&quot;language-python')