import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_dataframe_pane_supports_escape(document, comm):
    url = "<a href='https://panel.holoviz.org/'>Panel</a>"
    df = pd.DataFrame({'url': [url]})
    pane = DataFrame(df)
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert pane.escape
    assert '&lt;a href=&#x27;https://panel.holoviz.org/&#x27;&gt;Panel&lt;/a&gt;' not in model.text
    pane.escape = False
    assert '&lt;a href=&#x27;https://panel.holoviz.org/&#x27;&gt;Panel&lt;/a&gt;' in model.text
    pane._cleanup(model)
    assert pane._models == {}