from datetime import (
from decimal import Decimal
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
from pandas.core.arrays.string_arrow import ArrowStringArrayNumpySemantics
@pytest.mark.parametrize('op', ['min', 'max'])
def test_minmax_timedelta_empty_or_na(self, op):
    obj = TimedeltaIndex([])
    assert getattr(obj, op)() is NaT
    obj = TimedeltaIndex([NaT])
    assert getattr(obj, op)() is NaT
    obj = TimedeltaIndex([NaT, NaT, NaT])
    assert getattr(obj, op)() is NaT