import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_times_not_datetime_type():
    msg = 'times must be datetime64 dtype.'
    with pytest.raises(ValueError, match=msg):
        Series(range(5)).ewm(times=np.arange(5))