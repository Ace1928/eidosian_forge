import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_loc_get_scalar_casting_to_float():
    df = DataFrame({'a': 1.0, 'b': 2}, index=MultiIndex.from_arrays([[3], [4]], names=['c', 'd']))
    result = df.loc[(3, 4), 'b']
    assert result == 2
    assert isinstance(result, np.int64)
    result = df.loc[[(3, 4)], 'b'].iloc[0]
    assert result == 2
    assert isinstance(result, np.int64)