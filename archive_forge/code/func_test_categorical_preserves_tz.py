from __future__ import annotations
from datetime import (
from functools import partial
from operator import attrgetter
import dateutil
import dateutil.tz
from dateutil.tz import gettz
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import period_array
def test_categorical_preserves_tz(self):
    dti = DatetimeIndex([pd.NaT, '2015-01-01', '1999-04-06 15:14:13', '2015-01-01'], tz='US/Eastern')
    for dtobj in [dti, dti._data]:
        ci = pd.CategoricalIndex(dtobj)
        carr = pd.Categorical(dtobj)
        cser = pd.Series(ci)
        for obj in [ci, carr, cser]:
            result = DatetimeIndex(obj)
            tm.assert_index_equal(result, dti)