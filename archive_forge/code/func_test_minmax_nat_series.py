from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('nat_ser', [Series([NaT, NaT]), Series([NaT, Timedelta('nat')]), Series([Timedelta('nat'), Timedelta('nat')])])
def test_minmax_nat_series(self, nat_ser):
    assert nat_ser.min() is NaT
    assert nat_ser.max() is NaT
    assert nat_ser.min(skipna=False) is NaT
    assert nat_ser.max(skipna=False) is NaT