from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_arrays_tuples(idx):
    arrays = tuple((tuple(np.asarray(lev).take(level_codes)) for lev, level_codes in zip(idx.levels, idx.codes)))
    result = MultiIndex.from_arrays(arrays, names=idx.names)
    tm.assert_index_equal(result, idx)