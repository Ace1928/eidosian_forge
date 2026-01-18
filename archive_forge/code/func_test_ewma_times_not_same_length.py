import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_times_not_same_length():
    msg = 'times must be the same length as the object.'
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(4).astype('datetime64[ns]'))