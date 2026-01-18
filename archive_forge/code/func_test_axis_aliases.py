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
def test_axis_aliases(self, float_frame):
    f = float_frame
    expected = f.sum(axis=0)
    result = f.sum(axis='index')
    tm.assert_series_equal(result, expected)
    expected = f.sum(axis=1)
    result = f.sum(axis='columns')
    tm.assert_series_equal(result, expected)