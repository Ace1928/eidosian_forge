import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('side', ['right', 'left', 'both', 'neither'])
def test_get_loc_interval(self, closed, side):
    idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
    for bound in [[0, 1], [1, 2], [2, 3], [3, 4], [0, 2], [2.5, 3], [-1, 4]]:
        msg = re.escape(f"Interval({bound[0]}, {bound[1]}, closed='{side}')")
        if closed == side:
            if bound == [0, 1]:
                assert idx.get_loc(Interval(0, 1, closed=side)) == 0
            elif bound == [2, 3]:
                assert idx.get_loc(Interval(2, 3, closed=side)) == 1
            else:
                with pytest.raises(KeyError, match=msg):
                    idx.get_loc(Interval(*bound, closed=side))
        else:
            with pytest.raises(KeyError, match=msg):
                idx.get_loc(Interval(*bound, closed=side))