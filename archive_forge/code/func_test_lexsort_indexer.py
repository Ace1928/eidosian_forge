from collections import defaultdict
from datetime import datetime
from itertools import product
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
import pandas.core.common as com
from pandas.core.sorting import (
@pytest.mark.parametrize('order, na_position, exp', [[True, 'last', list(range(5, 105)) + list(range(5)) + list(range(105, 110))], [True, 'first', list(range(5)) + list(range(105, 110)) + list(range(5, 105))], [False, 'last', list(range(104, 4, -1)) + list(range(5)) + list(range(105, 110))], [False, 'first', list(range(5)) + list(range(105, 110)) + list(range(104, 4, -1))]])
def test_lexsort_indexer(self, order, na_position, exp):
    keys = [[np.nan] * 5 + list(range(100)) + [np.nan] * 5]
    result = lexsort_indexer(keys, orders=order, na_position=na_position)
    tm.assert_numpy_array_equal(result, np.array(exp, dtype=np.intp))