import numpy as np
import pytest
from pandas.core.dtypes.common import is_any_real_numeric_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_equals_ea_int_regular_int():
    mi1 = MultiIndex.from_arrays([Index([1, 2], dtype='Int64'), [3, 4]])
    mi2 = MultiIndex.from_arrays([[1, 2], [3, 4]])
    assert not mi1.equals(mi2)
    assert not mi2.equals(mi1)