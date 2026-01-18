import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
def test_json_applies():
    assert JSON.applies({1: 2})
    assert JSON.applies([1, 2, 3])
    assert JSON.applies('{"a": 1}') == 0
    assert not JSON.applies({'array': np.array([1, 2, 3])})
    assert JSON.applies({'array': np.array([1, 2, 3])}, encoder=NumpyEncoder)