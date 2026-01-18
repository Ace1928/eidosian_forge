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
@pytest.mark.parametrize(('keys', 'integrity'), [(['red'] * 3, True), (['red'] * 3, False), (['red', 'blue', 'red'], False), (['red', 'blue', 'red'], True)])
def test_concat_repeated_keys(keys, integrity):
    series_list = [Series({'a': 1}), Series({'b': 2}), Series({'c': 3})]
    result = concat(series_list, keys=keys, verify_integrity=integrity)
    tuples = list(zip(keys, ['a', 'b', 'c']))
    expected = Series([1, 2, 3], index=MultiIndex.from_tuples(tuples))
    tm.assert_series_equal(result, expected)