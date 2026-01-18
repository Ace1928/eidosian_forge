import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def test_ewma_times_adjust_false_raises():
    with pytest.raises(NotImplementedError, match='times is not supported with adjust=False.'):
        Series(range(1)).ewm(0.1, adjust=False, times=date_range('2000', freq='D', periods=1))