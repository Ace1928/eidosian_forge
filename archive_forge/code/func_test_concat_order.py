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
def test_concat_order(self):
    dfs = [DataFrame(index=range(3), columns=['a', 1, None])]
    dfs += [DataFrame(index=range(3), columns=[None, 1, 'a']) for _ in range(100)]
    result = concat(dfs, sort=True).columns
    expected = Index([1, 'a', None])
    tm.assert_index_equal(result, expected)