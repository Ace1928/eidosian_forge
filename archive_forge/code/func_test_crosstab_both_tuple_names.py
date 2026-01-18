import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_crosstab_both_tuple_names(self):
    s1 = Series(range(3), name=('a', 'b'))
    s2 = Series(range(3), name=('c', 'd'))
    expected = DataFrame(np.eye(3, dtype='int64'), index=Index(range(3), name=('a', 'b')), columns=Index(range(3), name=('c', 'd')))
    result = crosstab(s1, s2)
    tm.assert_frame_equal(result, expected)