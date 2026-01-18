import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_dataframe_pane_pandas(document, comm):
    pane = DataFrame(pd.DataFrame({'A': [1, 2, 3]}))
    model = pane.get_root(document, comm=comm)
    assert pane._models[model.ref['id']][0] is model
    assert model.text.startswith('&lt;table')
    orig_text = model.text
    pane.object = pd.DataFrame({'B': [1, 2, 3]})
    assert pane._models[model.ref['id']][0] is model
    assert model.text.startswith('&lt;table')
    assert model.text != orig_text
    pane._cleanup(model)
    assert pane._models == {}