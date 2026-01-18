import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_assign_order(self):
    df = DataFrame([[1, 2], [3, 4]], columns=['A', 'B'])
    result = df.assign(D=df.A + df.B, C=df.A - df.B)
    expected = DataFrame([[1, 2, 3, -1], [3, 4, 7, -1]], columns=list('ABDC'))
    tm.assert_frame_equal(result, expected)
    result = df.assign(C=df.A - df.B, D=df.A + df.B)
    expected = DataFrame([[1, 2, -1, 3], [3, 4, -1, 7]], columns=list('ABCD'))
    tm.assert_frame_equal(result, expected)