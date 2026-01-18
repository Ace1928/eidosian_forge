import operator
import numpy as np
import pytest
from pandas._libs.tslibs import (
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import ops
from pandas.core.arrays import TimedeltaArray
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [pd.date_range('2000', periods=4).array, pd.timedelta_range('1D', periods=4).array, np.arange(4), np.arange(4).astype(np.float64), list(range(4)), [2000, 2001, 2002, 2003], np.arange(2000, 2004), np.arange(2000, 2004).astype(object), pd.Index([2000, 2001, 2002, 2003])])
def test_compare_invalid_listlike(self, box_with_array, other):
    pi = period_range('2000', periods=4)
    parr = tm.box_expected(pi, box_with_array)
    assert_invalid_comparison(parr, other, box_with_array)