from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_arrays_empty():
    msg = 'Must pass non-zero number of levels/codes'
    with pytest.raises(ValueError, match=msg):
        MultiIndex.from_arrays(arrays=[])
    result = MultiIndex.from_arrays(arrays=[[]], names=['A'])
    assert isinstance(result, MultiIndex)
    expected = Index([], name='A')
    tm.assert_index_equal(result.levels[0], expected)
    assert result.names == ['A']
    for N in [2, 3]:
        arrays = [[]] * N
        names = list('ABC')[:N]
        result = MultiIndex.from_arrays(arrays=arrays, names=names)
        expected = MultiIndex(levels=[[]] * N, codes=[[]] * N, names=names)
        tm.assert_index_equal(result, expected)