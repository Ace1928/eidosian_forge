from datetime import (
import operator
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p24p3
from pandas import (
import pandas._testing as tm
from pandas.core import roperator
from pandas.core.arrays import (
@pytest.mark.parametrize('other', [np.timedelta64(0, 'ns'), np.datetime64('now', 'ns')])
def test_nat_comparisons_numpy(other):
    assert not NaT == other
    assert NaT != other
    assert not NaT < other
    assert not NaT > other
    assert not NaT <= other
    assert not NaT >= other