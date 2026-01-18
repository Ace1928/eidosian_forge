import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_concat_bool_with_int(self):
    df1 = DataFrame(Series([True, False, True, True], dtype='bool'))
    df2 = DataFrame(Series([1, 0, 1], dtype='int64'))
    result = concat([df1, df2])
    expected = concat([df1.astype('int64'), df2])
    tm.assert_frame_equal(result, expected)