from datetime import (
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multilevel_name_print_0(self):
    mi = pd.MultiIndex.from_product([range(2, 3), range(3, 4)], names=[0, None])
    ser = Series(1.5, index=mi)
    res = repr(ser)
    expected = '0   \n2  3    1.5\ndtype: float64'
    assert res == expected