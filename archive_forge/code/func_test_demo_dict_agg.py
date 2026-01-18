from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_demo_dict_agg():
    df = DataFrame({'A': range(5), 'B': 5})
    result = df.agg({'A': ['min', 'max'], 'B': ['sum', 'max']})
    expected = DataFrame({'A': [4.0, 0.0, np.nan], 'B': [5.0, np.nan, 25.0]}, columns=['A', 'B'], index=['max', 'min', 'sum'])
    tm.assert_frame_equal(result.reindex_like(expected), expected)