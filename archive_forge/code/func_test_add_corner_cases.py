from datetime import (
from decimal import Decimal
import operator
import numpy as np
import pytest
from pandas._libs import lib
from pandas._libs.tslibs import IncompatibleFrequency
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.computation import expressions as expr
from pandas.core.computation.check import NUMEXPR_INSTALLED
def test_add_corner_cases(self, datetime_series):
    empty = Series([], index=Index([]), dtype=np.float64)
    result = datetime_series + empty
    assert np.isnan(result).all()
    result = empty + empty.copy()
    assert len(result) == 0