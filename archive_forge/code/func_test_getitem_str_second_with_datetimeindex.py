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
def test_getitem_str_second_with_datetimeindex():
    df = DataFrame(np.random.default_rng(2).random((5, 5)), columns=['open', 'high', 'low', 'close', 'volume'], index=date_range('2012-01-02 18:01:00', periods=5, tz='US/Central', freq='s'))
    with pytest.raises(KeyError, match="^'2012-01-02 18:01:02'$"):
        df['2012-01-02 18:01:02']
    msg = "Timestamp\\('2012-01-02 18:01:02-0600', tz='US/Central'\\)"
    with pytest.raises(KeyError, match=msg):
        df[df.index[2]]