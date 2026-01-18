import numpy as np
import pytest
from pandas._libs.tslibs.period import IncompatibleFrequency
from pandas.core.dtypes.dtypes import PeriodDtype
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import PeriodArray
def test_keyword_mismatch(self):
    per = Period('2016-01-01', 'D')
    depr_msg1 = "The 'ordinal' keyword in PeriodIndex is deprecated"
    depr_msg2 = 'Constructing PeriodIndex from fields is deprecated'
    err_msg1 = 'Cannot pass both data and ordinal'
    with pytest.raises(ValueError, match=err_msg1):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg1):
            PeriodIndex(data=[per], ordinal=[per.ordinal], freq=per.freq)
    err_msg2 = 'Cannot pass both data and fields'
    with pytest.raises(ValueError, match=err_msg2):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg2):
            PeriodIndex(data=[per], year=[per.year], freq=per.freq)
    err_msg3 = 'Cannot pass both ordinal and fields'
    with pytest.raises(ValueError, match=err_msg3):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg2):
            PeriodIndex(ordinal=[per.ordinal], year=[per.year], freq=per.freq)