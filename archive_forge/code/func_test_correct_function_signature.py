import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def test_correct_function_signature():
    pytest.importorskip('numba')

    def incorrect_function(x):
        return x + 1
    data = DataFrame({'key': ['a', 'a', 'b', 'b', 'a'], 'data': [1.0, 2.0, 3.0, 4.0, 5.0]}, columns=['key', 'data'])
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key').transform(incorrect_function, engine='numba')
    with pytest.raises(NumbaUtilError, match='The first 2'):
        data.groupby('key')['data'].transform(incorrect_function, engine='numba')