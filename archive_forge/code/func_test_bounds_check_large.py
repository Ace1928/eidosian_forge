from datetime import datetime
import numpy as np
import pytest
from pandas._libs import iNaT
import pandas._testing as tm
import pandas.core.algorithms as algos
def test_bounds_check_large(self):
    arr = np.array([1, 2])
    msg = 'indices are out-of-bounds'
    with pytest.raises(IndexError, match=msg):
        algos.take(arr, [2, 3], allow_fill=True)
    msg = 'index 2 is out of bounds for( axis 0 with)? size 2'
    with pytest.raises(IndexError, match=msg):
        algos.take(arr, [2, 3], allow_fill=False)