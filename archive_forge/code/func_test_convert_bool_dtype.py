from itertools import product
import numpy as np
import pytest
from pandas._libs import lib
import pandas as pd
import pandas._testing as tm
def test_convert_bool_dtype(self):
    df = pd.DataFrame({'A': pd.array([True])})
    tm.assert_frame_equal(df, df.convert_dtypes())