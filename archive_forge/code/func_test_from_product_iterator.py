from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_product_iterator():
    first = ['foo', 'bar', 'buz']
    second = ['a', 'b', 'c']
    names = ['first', 'second']
    tuples = [('foo', 'a'), ('foo', 'b'), ('foo', 'c'), ('bar', 'a'), ('bar', 'b'), ('bar', 'c'), ('buz', 'a'), ('buz', 'b'), ('buz', 'c')]
    expected = MultiIndex.from_tuples(tuples, names=names)
    result = MultiIndex.from_product(iter([first, second]), names=names)
    tm.assert_index_equal(result, expected)
    msg = 'Input must be a list / sequence of iterables.'
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_product(0)