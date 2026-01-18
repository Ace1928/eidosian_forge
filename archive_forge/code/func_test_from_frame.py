from datetime import (
import itertools
import numpy as np
import pytest
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_from_frame():
    df = pd.DataFrame([['a', 'a'], ['a', 'b'], ['b', 'a'], ['b', 'b']], columns=['L1', 'L2'])
    expected = MultiIndex.from_tuples([('a', 'a'), ('a', 'b'), ('b', 'a'), ('b', 'b')], names=['L1', 'L2'])
    result = MultiIndex.from_frame(df)
    tm.assert_index_equal(expected, result)