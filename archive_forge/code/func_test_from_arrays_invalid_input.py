from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.mark.parametrize('invalid_sequence_of_arrays', [1, [1], [1, 2], [[1], 2], [1, [2]], 'a', ['a'], ['a', 'b'], [['a'], 'b'], (1,), (1, 2), ([1], 2), (1, [2]), 'a', ('a',), ('a', 'b'), (['a'], 'b'), [(1,), 2], [1, (2,)], [('a',), 'b'], ((1,), 2), (1, (2,)), (('a',), 'b')])
def test_from_arrays_invalid_input(invalid_sequence_of_arrays):
    msg = 'Input must be a list / sequence of array-likes'
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(arrays=invalid_sequence_of_arrays)