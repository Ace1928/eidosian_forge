import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('scalar', [-1, 0, 0.5, 3, 4.5, 5, 6])
def test_get_loc_length_one_scalar(self, scalar, closed):
    index = IntervalIndex.from_tuples([(0, 5)], closed=closed)
    if scalar in index[0]:
        result = index.get_loc(scalar)
        assert result == 0
    else:
        with pytest.raises(KeyError, match=str(scalar)):
            index.get_loc(scalar)