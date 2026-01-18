import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('scalar', [-0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5])
def test_get_loc_scalar(self, closed, scalar):
    correct = {'right': {0.5: 0, 1: 0, 2.5: 1, 3: 1}, 'left': {0: 0, 0.5: 0, 2: 1, 2.5: 1}, 'both': {0: 0, 0.5: 0, 1: 0, 2: 1, 2.5: 1, 3: 1}, 'neither': {0.5: 0, 2.5: 1}}
    idx = IntervalIndex.from_tuples([(0, 1), (2, 3)], closed=closed)
    if scalar in correct[closed].keys():
        assert idx.get_loc(scalar) == correct[closed][scalar]
    else:
        with pytest.raises(KeyError, match=str(scalar)):
            idx.get_loc(scalar)