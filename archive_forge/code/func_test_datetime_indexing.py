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
def test_datetime_indexing():
    index = date_range('1/1/2000', '1/7/2000')
    index = index.repeat(3)
    s = Series(len(index), index=index)
    stamp = Timestamp('1/8/2000')
    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    s[stamp] = 0
    assert s[stamp] == 0
    s = Series(len(index), index=index)
    s = s[::-1]
    with pytest.raises(KeyError, match=re.escape(repr(stamp))):
        s[stamp]
    s[stamp] = 0
    assert s[stamp] == 0