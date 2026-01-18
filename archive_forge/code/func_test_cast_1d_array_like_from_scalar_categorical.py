import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_arraylike_from_scalar
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_cast_1d_array_like_from_scalar_categorical():
    cats = ['a', 'b', 'c']
    cat_type = CategoricalDtype(categories=cats, ordered=False)
    expected = Categorical(['a', 'a'], categories=cats)
    result = construct_1d_arraylike_from_scalar('a', len(expected), cat_type)
    tm.assert_categorical_equal(result, expected)