from datetime import (
from itertools import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.conversion import localize_pydatetime
from pandas._libs.tslibs.offsets import shift_months
from pandas.errors import PerformanceWarning
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.tests.arithmetic.common import (
@pytest.mark.parametrize('other', [3.14, np.array([2.0, 3.0]), Period('2011-01-01', freq='D'), time(1, 2, 3)])
@pytest.mark.parametrize('dti_freq', [None, 'D'])
def test_dt64arr_add_sub_invalid(self, dti_freq, other, box_with_array):
    dti = DatetimeIndex(['2011-01-01', '2011-01-02'], freq=dti_freq)
    dtarr = tm.box_expected(dti, box_with_array)
    msg = '|'.join(['unsupported operand type', 'cannot (add|subtract)', 'cannot use operands with types', "ufunc '?(add|subtract)'? cannot use operands with types", 'Concatenation operation is not implemented for NumPy arrays'])
    assert_invalid_addsub_type(dtarr, other, msg)