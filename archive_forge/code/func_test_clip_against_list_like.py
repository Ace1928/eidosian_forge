import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('inplace', [True, False])
@pytest.mark.parametrize('lower', [[2, 3, 4], np.asarray([2, 3, 4])])
@pytest.mark.parametrize('axis,res', [(0, [[2.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 7.0, 7.0]]), (1, [[2.0, 3.0, 4.0], [4.0, 5.0, 6.0], [5.0, 6.0, 7.0]])])
def test_clip_against_list_like(self, inplace, lower, axis, res):
    arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    original = DataFrame(arr, columns=['one', 'two', 'three'], index=['a', 'b', 'c'])
    result = original.clip(lower=lower, upper=[5, 6, 7], axis=axis, inplace=inplace)
    expected = DataFrame(res, columns=original.columns, index=original.index)
    if inplace:
        result = original
    tm.assert_frame_equal(result, expected, check_exact=True)