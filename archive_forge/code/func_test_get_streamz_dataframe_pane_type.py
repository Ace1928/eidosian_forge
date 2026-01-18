import base64
import json
import numpy as np
import pandas as pd
from panel.pane import (
from panel.tests.util import streamz_available
@streamz_available
def test_get_streamz_dataframe_pane_type():
    from streamz.dataframe import Random
    sdf = Random(interval='200ms', freq='50ms')
    assert PaneBase.get_pane_type(sdf) is DataFrame