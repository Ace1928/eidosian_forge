from datetime import (
import re
import numpy as np
import pytest
from pandas._libs.tslibs import iNaT
from pandas._libs.tslibs.ccalendar import (
from pandas._libs.tslibs.np_datetime import OutOfBoundsDatetime
from pandas._libs.tslibs.parsing import DateParseError
from pandas._libs.tslibs.period import INVALID_FREQ_ERR_MSG
from pandas import (
import pandas._testing as tm
def test_custom_business_day_freq_raises(self):
    msg = 'C is not supported as period frequency'
    with pytest.raises(ValueError, match=msg):
        Period('2023-04-10', freq='C')
    msg = f'{offsets.CustomBusinessDay().base} is not supported as period frequency'
    with pytest.raises(ValueError, match=msg):
        Period('2023-04-10', freq=offsets.CustomBusinessDay())