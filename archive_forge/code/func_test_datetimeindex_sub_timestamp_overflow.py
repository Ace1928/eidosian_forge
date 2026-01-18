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
def test_datetimeindex_sub_timestamp_overflow(self):
    dtimax = pd.to_datetime(['2021-12-28 17:19', Timestamp.max]).as_unit('ns')
    dtimin = pd.to_datetime(['2021-12-28 17:19', Timestamp.min]).as_unit('ns')
    tsneg = Timestamp('1950-01-01').as_unit('ns')
    ts_neg_variants = [tsneg, tsneg.to_pydatetime(), tsneg.to_datetime64().astype('datetime64[ns]'), tsneg.to_datetime64().astype('datetime64[D]')]
    tspos = Timestamp('1980-01-01').as_unit('ns')
    ts_pos_variants = [tspos, tspos.to_pydatetime(), tspos.to_datetime64().astype('datetime64[ns]'), tspos.to_datetime64().astype('datetime64[D]')]
    msg = 'Overflow in int64 addition'
    for variant in ts_neg_variants:
        with pytest.raises(OverflowError, match=msg):
            dtimax - variant
    expected = Timestamp.max._value - tspos._value
    for variant in ts_pos_variants:
        res = dtimax - variant
        assert res[1]._value == expected
    expected = Timestamp.min._value - tsneg._value
    for variant in ts_neg_variants:
        res = dtimin - variant
        assert res[1]._value == expected
    for variant in ts_pos_variants:
        with pytest.raises(OverflowError, match=msg):
            dtimin - variant