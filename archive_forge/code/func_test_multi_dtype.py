import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multi_dtype(self):
    df = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=['a', 'a', 'b', 'b', 'd', 'c', 'c'])
    df.columns = list('ABCDEFG')
    expected = DataFrame([[1, 2, 1.0, 2.0, 3.0, 'foo', 'bar']], columns=list('ABCDEFG'))
    tm.assert_frame_equal(df, expected)