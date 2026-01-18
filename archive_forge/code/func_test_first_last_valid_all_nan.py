import numpy as np
import pytest
from pandas import (
@pytest.mark.parametrize('index', [Index([str(i) for i in range(10)]), date_range('2020-01-01', periods=10)])
def test_first_last_valid_all_nan(self, index):
    frame = DataFrame(np.nan, columns=['foo'], index=index)
    assert frame.last_valid_index() is None
    assert frame.first_valid_index() is None
    ser = frame['foo']
    assert ser.first_valid_index() is None
    assert ser.last_valid_index() is None