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
def test_concat_with_group_keys(self):
    df = DataFrame(np.random.default_rng(2).standard_normal((3, 4)))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))
    result = concat([df, df2], keys=[0, 1])
    exp_index = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1, 1], [0, 1, 2, 0, 1, 2, 3]])
    expected = DataFrame(np.r_[df.values, df2.values], index=exp_index)
    tm.assert_frame_equal(result, expected)
    result = concat([df, df], keys=[0, 1])
    exp_index2 = MultiIndex.from_arrays([[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]])
    expected = DataFrame(np.r_[df.values, df.values], index=exp_index2)
    tm.assert_frame_equal(result, expected)
    df = DataFrame(np.random.default_rng(2).standard_normal((4, 3)))
    df2 = DataFrame(np.random.default_rng(2).standard_normal((4, 4)))
    result = concat([df, df2], keys=[0, 1], axis=1)
    expected = DataFrame(np.c_[df.values, df2.values], columns=exp_index)
    tm.assert_frame_equal(result, expected)
    result = concat([df, df], keys=[0, 1], axis=1)
    expected = DataFrame(np.c_[df.values, df.values], columns=exp_index2)
    tm.assert_frame_equal(result, expected)