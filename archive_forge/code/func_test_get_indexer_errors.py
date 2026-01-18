import re
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('tuples, closed', [([(0, 2), (1, 3), (3, 4)], 'neither'), ([(0, 5), (1, 4), (6, 7)], 'left'), ([(0, 1), (0, 1), (1, 2)], 'right'), ([(0, 1), (2, 3), (3, 4)], 'both')])
def test_get_indexer_errors(self, tuples, closed):
    index = IntervalIndex.from_tuples(tuples, closed=closed)
    msg = 'cannot handle overlapping indices; use IntervalIndex.get_indexer_non_unique'
    with pytest.raises(InvalidIndexError, match=msg):
        index.get_indexer([0, 2])