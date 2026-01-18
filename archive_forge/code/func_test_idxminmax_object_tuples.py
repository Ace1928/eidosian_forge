from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_idxminmax_object_tuples(self):
    ser = Series([(1, 3), (2, 2), (3, 1)])
    assert ser.idxmax() == 2
    assert ser.idxmin() == 0
    assert ser.idxmax(skipna=False) == 2
    assert ser.idxmin(skipna=False) == 0