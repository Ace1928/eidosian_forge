from datetime import datetime
import warnings
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.frame.common import zip_frames
def test_demo():
    df = DataFrame({'A': range(5), 'B': 5})
    result = df.agg(['min', 'max'])
    expected = DataFrame({'A': [0, 4], 'B': [5, 5]}, columns=['A', 'B'], index=['min', 'max'])
    tm.assert_frame_equal(result, expected)