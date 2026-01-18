import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas._libs.tslibs.period import (
import pandas._testing as tm
def test_get_period_field_array_raises_on_out_of_range():
    msg = "Buffer dtype mismatch, expected 'const int64_t' but got 'double'"
    with pytest.raises(ValueError, match=msg):
        get_period_field_arr(-1, np.empty(1), 0)