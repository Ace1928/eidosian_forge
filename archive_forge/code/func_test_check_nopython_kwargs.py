import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def test_check_nopython_kwargs():
    pytest.importorskip('numba')

    def incorrect_function(values, index):
        return values + 1
    data = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key').transform(incorrect_function, engine='numba', a=1)
    with pytest.raises(NumbaUtilError, match='numba does not support'):
        data.groupby('key')['data'].transform(incorrect_function, engine='numba', a=1)