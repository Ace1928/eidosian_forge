from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dtype_representation(using_infer_string):
    pmidx = MultiIndex.from_arrays([[1], ['a']], names=[('a', 'b'), ('c', 'd')])
    result = pmidx.dtypes
    exp = 'object' if not using_infer_string else 'string'
    expected = Series(['int64', exp], index=MultiIndex.from_tuples([('a', 'b'), ('c', 'd')]), dtype=object)
    tm.assert_series_equal(result, expected)