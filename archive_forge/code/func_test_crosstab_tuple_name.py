import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('names', [['a', ('b', 'c')], [('a', 'b'), 'c']])
def test_crosstab_tuple_name(self, names):
    s1 = Series(range(3), name=names[0])
    s2 = Series(range(1, 4), name=names[1])
    mi = MultiIndex.from_arrays([range(3), range(1, 4)], names=names)
    expected = Series(1, index=mi).unstack(1, fill_value=0)
    result = crosstab(s1, s2)
    tm.assert_frame_equal(result, expected)