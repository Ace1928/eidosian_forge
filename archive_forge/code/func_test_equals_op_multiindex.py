from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('mi,expected', [(MultiIndex.from_tuples([(1, 2), (4, 5)]), np.array([True, True])), (MultiIndex.from_tuples([(1, 2), (4, 6)]), np.array([True, False]))])
def test_equals_op_multiindex(self, mi, expected):
    df = DataFrame([3, 6], columns=['c'], index=MultiIndex.from_arrays([[1, 4], [2, 5]], names=['a', 'b']))
    result = df.index == mi
    tm.assert_numpy_array_equal(result, expected)