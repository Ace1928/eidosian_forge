from contextlib import nullcontext
from datetime import datetime
from decimal import Decimal
import numpy as np
import pytest
from pandas._config import config as cf
from pandas._libs import missing as libmissing
from pandas._libs.tslibs import iNaT
from pandas.compat.numpy import np_version_gte1p25
from pandas.core.dtypes.common import (
from pandas.core.dtypes.dtypes import (
from pandas.core.dtypes.missing import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_isna_old_datetimelike(self):
    dti = date_range('2016-01-01', periods=3)
    dta = dti._data
    dta[-1] = NaT
    expected = np.array([False, False, True], dtype=bool)
    objs = [dta, dta.tz_localize('US/Eastern'), dta - dta, dta.to_period('D')]
    for obj in objs:
        msg = 'use_inf_as_na option is deprecated'
        with tm.assert_produces_warning(FutureWarning, match=msg):
            with cf.option_context('mode.use_inf_as_na', True):
                result = isna(obj)
        tm.assert_numpy_array_equal(result, expected)