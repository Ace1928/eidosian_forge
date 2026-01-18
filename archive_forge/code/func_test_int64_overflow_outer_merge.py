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
def test_int64_overflow_outer_merge(self):
    df1 = DataFrame(np.random.default_rng(2).standard_normal((1000, 7)), columns=list('ABCDEF') + ['G1'])
    df2 = DataFrame(np.random.default_rng(3).standard_normal((1000, 7)), columns=list('ABCDEF') + ['G2'])
    result = merge(df1, df2, how='outer')
    assert len(result) == 2000