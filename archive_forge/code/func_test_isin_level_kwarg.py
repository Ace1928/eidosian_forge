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
@pytest.mark.parametrize('level', [0, -1])
@pytest.mark.parametrize('index', [Index(['qux', 'baz', 'foo', 'bar']), Index([1.0, 2.0, 3.0, 4.0], dtype=np.float64)])
def test_isin_level_kwarg(self, level, index):
    values = index.tolist()[-2:] + ['nonexisting']
    expected = np.array([False, False, True, True])
    tm.assert_numpy_array_equal(expected, index.isin(values, level=level))
    index.name = 'foobar'
    tm.assert_numpy_array_equal(expected, index.isin(values, level='foobar'))