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
def test_from_td64nat_raises(self):
    td = NaT.to_numpy('m8[ns]')
    msg = 'Value must be Period, string, integer, or datetime'
    with pytest.raises(ValueError, match=msg):
        Period(td)
    with pytest.raises(ValueError, match=msg):
        Period(td, freq='D')