import pytest
from pandas import DataFrame
import pandas._testing as tm
def test_assign_multiple(self):
    df = DataFrame([[1, 4], [2, 5], [3, 6]], columns=['A', 'B'])
    result = df.assign(C=[7, 8, 9], D=df.A, E=lambda x: x.B)
    expected = DataFrame([[1, 4, 7, 1, 4], [2, 5, 8, 2, 5], [3, 6, 9, 3, 6]], columns=list('ABCDE'))
    tm.assert_frame_equal(result, expected)