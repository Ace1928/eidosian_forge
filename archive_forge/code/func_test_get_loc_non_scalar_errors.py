import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('key', [[5], (2, 3)])
def test_get_loc_non_scalar_errors(self, key):
    idx = IntervalIndex.from_tuples([(1, 3), (2, 4), (3, 5), (7, 10), (3, 10)])
    msg = str(key)
    with pytest.raises(InvalidIndexError, match=msg):
        idx.get_loc(key)