import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_get_series_pane_type():
    ser = pd.Series([1, 2, 3])
    assert PaneBase.get_pane_type(ser) is DataFrame