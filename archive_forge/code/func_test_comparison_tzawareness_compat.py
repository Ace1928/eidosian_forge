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
def test_comparison_tzawareness_compat(self, comparison_op, box_with_array):
    op = comparison_op
    box = box_with_array
    dr = date_range('2016-01-01', periods=6)
    dz = dr.tz_localize('US/Pacific')
    dr = tm.box_expected(dr, box)
    dz = tm.box_expected(dz, box)
    if box is pd.DataFrame:
        tolist = lambda x: x.astype(object).values.tolist()[0]
    else:
        tolist = list
    if op not in [operator.eq, operator.ne]:
        msg = 'Invalid comparison between dtype=datetime64\\[ns.*\\] and (Timestamp|DatetimeArray|list|ndarray)'
        with pytest.raises(TypeError, match=msg):
            op(dr, dz)
        with pytest.raises(TypeError, match=msg):
            op(dr, tolist(dz))
        with pytest.raises(TypeError, match=msg):
            op(dr, np.array(tolist(dz), dtype=object))
        with pytest.raises(TypeError, match=msg):
            op(dz, dr)
        with pytest.raises(TypeError, match=msg):
            op(dz, tolist(dr))
        with pytest.raises(TypeError, match=msg):
            op(dz, np.array(tolist(dr), dtype=object))
    assert np.all(dr == dr)
    assert np.all(dr == tolist(dr))
    assert np.all(tolist(dr) == dr)
    assert np.all(np.array(tolist(dr), dtype=object) == dr)
    assert np.all(dr == np.array(tolist(dr), dtype=object))
    assert np.all(dz == dz)
    assert np.all(dz == tolist(dz))
    assert np.all(tolist(dz) == dz)
    assert np.all(np.array(tolist(dz), dtype=object) == dz)
    assert np.all(dz == np.array(tolist(dz), dtype=object))