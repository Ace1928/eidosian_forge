from copy import deepcopy
import inspect
import pydoc
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._config.config import option_context
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_get_agg_axis(self, float_frame):
    cols = float_frame._get_agg_axis(0)
    assert cols is float_frame.columns
    idx = float_frame._get_agg_axis(1)
    assert idx is float_frame.index
    msg = 'Axis must be 0 or 1 \\(got 2\\)'
    with pytest.raises(ValueError, match=msg):
        float_frame._get_agg_axis(2)