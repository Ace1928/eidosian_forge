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
def test_inv(self, simple_index):
    idx = simple_index
    if idx.dtype.kind in ['i', 'u']:
        res = ~idx
        expected = Index(~idx.values, name=idx.name)
        tm.assert_index_equal(res, expected)
        res2 = ~Series(idx)
        tm.assert_series_equal(res2, Series(expected))
    else:
        if idx.dtype.kind == 'f':
            msg = "ufunc 'invert' not supported for the input types"
        else:
            msg = 'bad operand'
        with pytest.raises(TypeError, match=msg):
            ~idx
        with pytest.raises(TypeError, match=msg):
            ~Series(idx)