from collections import (
from collections.abc import Iterator
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
from pandas.tests.extension.decimal import to_decimal
def test_concat_keys_specific_levels(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((10, 4)))
    pieces = [df.iloc[:, [0, 1]], df.iloc[:, [2]], df.iloc[:, [3]]]
    level = ['three', 'two', 'one', 'zero']
    result = concat(pieces, axis=1, keys=['one', 'two', 'three'], levels=[level], names=['group_key'])
    tm.assert_index_equal(result.columns.levels[0], Index(level, name='group_key'))
    tm.assert_index_equal(result.columns.levels[1], Index([0, 1, 2, 3]))
    assert result.columns.names == ['group_key', None]