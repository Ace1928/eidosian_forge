import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multi_dtype2(self):
    df = DataFrame([[1, 2, 'foo', 'bar']], columns=['a', 'a', 'a', 'a'])
    df.columns = ['a', 'a.1', 'a.2', 'a.3']
    expected = DataFrame([[1, 2, 'foo', 'bar']], columns=['a', 'a.1', 'a.2', 'a.3'])
    tm.assert_frame_equal(df, expected)