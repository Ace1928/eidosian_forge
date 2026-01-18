from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_arrays(idx):
    arrays = [np.asarray(lev).take(level_codes) for lev, level_codes in zip(idx.levels, idx.codes)]
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)
    result = MultiIndex.from_arrays([[pd.NaT, Timestamp('20130101')], ['a', 'b']])
    assert result.levels[0].equals(Index([Timestamp('20130101')]))
    assert result.levels[1].equals(Index(['a', 'b']))