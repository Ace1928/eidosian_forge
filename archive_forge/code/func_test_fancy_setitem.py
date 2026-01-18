from datetime import (
import re
from dateutil.tz import (
import numpy as np
import pytest
import pytz
from pandas._libs import index as libindex
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_fancy_setitem():
    dti = date_range(freq='WOM-1FRI', start=datetime(2005, 1, 1), end=datetime(2010, 1, 1))
    s = Series(np.arange(len(dti)), index=dti)
    msg = 'Series.__setitem__ treating keys as positions is deprecated'
    with tm.assert_produces_warning(FutureWarning, match=msg):
        s[48] = -1
    assert s.iloc[48] == -1
    s['1/2/2009'] = -2
    assert s.iloc[48] == -2
    s['1/2/2009':'2009-06-05'] = -3
    assert (s[48:54] == -3).all()