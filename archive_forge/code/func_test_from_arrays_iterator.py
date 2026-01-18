from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_arrays_iterator(idx):
    arrays = [np.asarray(lev).take(level_codes) for lev, level_codes in zip(idx.levels, idx.codes)]
    result = MultiIndex.from_arrays(iter(arrays), names=idx.names)
    tm.assert_index_equal(result, idx)
    msg = 'Input must be a list / sequence of array-likes.'
    with pytest.raises(TypeError, match=msg):
        MultiIndex.from_arrays(0)