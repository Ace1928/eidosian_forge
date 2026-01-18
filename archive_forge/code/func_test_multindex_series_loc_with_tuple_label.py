import numpy as np
import pytest
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_multindex_series_loc_with_tuple_label():
    mi = MultiIndex.from_tuples([(1, 2), (3, (4, 5))])
    ser = Series([1, 2], index=mi)
    result = ser.loc[3, (4, 5)]
    assert result == 2