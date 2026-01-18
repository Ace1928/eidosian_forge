from datetime import timedelta
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays.timedeltas import TimedeltaArray
def test_infer_from_tdi_mismatch(self):
    tdi = timedelta_range('1 second', periods=100, freq='1s')
    depr_msg = 'TimedeltaArray.__init__ is deprecated'
    msg = 'Inferred frequency .* from passed values does not conform to passed frequency'
    with pytest.raises(ValueError, match=msg):
        TimedeltaIndex(tdi, freq='D')
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            TimedeltaArray(tdi, freq='D')
    with pytest.raises(ValueError, match=msg):
        TimedeltaIndex(tdi._data, freq='D')
    with pytest.raises(ValueError, match=msg):
        with tm.assert_produces_warning(FutureWarning, match=depr_msg):
            TimedeltaArray(tdi._data, freq='D')