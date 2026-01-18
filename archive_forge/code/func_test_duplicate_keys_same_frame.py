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
def test_duplicate_keys_same_frame():
    keys = ['e', 'e']
    df = DataFrame({'a': [1, 2, 3], 'b': [4, 5, 6]})
    result = concat([df, df], axis=1, keys=keys)
    expected_values = [[1, 4, 1, 4], [2, 5, 2, 5], [3, 6, 3, 6]]
    expected_columns = MultiIndex.from_tuples([(keys[0], 'a'), (keys[0], 'b'), (keys[1], 'a'), (keys[1], 'b')])
    expected = DataFrame(expected_values, columns=expected_columns)
    tm.assert_frame_equal(result, expected)