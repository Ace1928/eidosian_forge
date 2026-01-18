import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas.errors import ChainedAssignmentError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
def test_interp_raise_on_only_mixed(self, axis):
    df = DataFrame({'A': [1, 2, np.nan, 4], 'B': ['a', 'b', 'c', 'd'], 'C': [np.nan, 2, 5, 7], 'D': [np.nan, np.nan, 9, 9], 'E': [1, 2, 3, 4]})
    msg = 'Cannot interpolate with all object-dtype columns in the DataFrame. Try setting at least one column to a numeric dtype.'
    with pytest.raises(TypeError, match=msg):
        df.astype('object').interpolate(axis=axis)