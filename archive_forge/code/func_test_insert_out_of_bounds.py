from __future__ import annotations
from datetime import datetime
import gc
import numpy as np
import pytest
from pandas._libs.tslibs import Timestamp
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import CategoricalDtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import BaseMaskedArray
def test_insert_out_of_bounds(self, index):
    if len(index) > 0:
        err = TypeError
    else:
        err = IndexError
    if len(index) == 0:
        msg = 'index (0|0.5) is out of bounds for axis 0 with size 0'
    else:
        msg = 'slice indices must be integers or None or have an __index__ method'
    with pytest.raises(err, match=msg):
        index.insert(0.5, 'foo')
    msg = '|'.join(['index -?\\d+ is out of bounds for axis 0 with size \\d+', 'loc must be an integer between'])
    with pytest.raises(IndexError, match=msg):
        index.insert(len(index) + 1, 1)
    with pytest.raises(IndexError, match=msg):
        index.insert(-len(index) - 1, 1)