import re
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import SparseArray
def test_getitem_list_missing_key(self):
    df = DataFrame({'x': [1.0], 'y': [2.0], 'z': [3.0]})
    df.columns = ['x', 'x', 'z']
    with pytest.raises(KeyError, match="\\['y'\\] not in index"):
        df[['x', 'y', 'z']]