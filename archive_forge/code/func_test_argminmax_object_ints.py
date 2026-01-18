from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
def test_argminmax_object_ints(self):
    ser = Series([0, 1], dtype='object')
    assert ser.argmax() == 1
    assert ser.argmin() == 0
    assert ser.argmax(skipna=False) == 1
    assert ser.argmin(skipna=False) == 0