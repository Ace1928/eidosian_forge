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
@pytest.mark.parametrize('nat,idx', [(Timestamp('NaT'), DatetimeArray), (Timedelta('NaT'), TimedeltaArray), (Period('NaT', freq='M'), PeriodArray)])
def test_nat_fields(nat, idx):
    for field in idx._field_ops:
        if field == 'weekday':
            continue
        result = getattr(NaT, field)
        assert np.isnan(result)
        result = getattr(nat, field)
        assert np.isnan(result)
    for field in idx._bool_ops:
        result = getattr(NaT, field)
        assert result is False
        result = getattr(nat, field)
        assert result is False