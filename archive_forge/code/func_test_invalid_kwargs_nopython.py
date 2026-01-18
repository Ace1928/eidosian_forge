import numpy as np
import pytest
from pandas.errors import NumbaUtilError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@td.skip_if_no('numba')
def test_invalid_kwargs_nopython():
    with pytest.raises(NumbaUtilError, match='numba does not support kwargs with'):
        Series(range(1)).rolling(1).apply(lambda x: x, kwargs={'a': 1}, engine='numba', raw=True)