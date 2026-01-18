import numpy as np
import pytest
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_getitem_boolean_ea_indexer():
    ser = pd.Series([True, False, pd.NA], dtype='boolean')
    result = ser.index[ser]
    expected = Index([0])
    tm.assert_index_equal(result, expected)