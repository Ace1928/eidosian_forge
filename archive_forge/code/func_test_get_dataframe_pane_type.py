import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_get_dataframe_pane_type():
    df = pd.DataFrame({'A': [1, 2, 3]})
    assert PaneBase.get_pane_type(df) is DataFrame