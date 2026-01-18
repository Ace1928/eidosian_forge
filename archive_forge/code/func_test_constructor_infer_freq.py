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
def test_constructor_infer_freq(self):
    p = Period('2007-01-01')
    assert p.freq == 'D'
    p = Period('2007-01-01 07')
    assert p.freq == 'h'
    p = Period('2007-01-01 07:10')
    assert p.freq == 'min'
    p = Period('2007-01-01 07:10:15')
    assert p.freq == 's'
    p = Period('2007-01-01 07:10:15.123')
    assert p.freq == 'ms'
    p = Period('2007-01-01 07:10:15.123000')
    assert p.freq == 'us'
    p = Period('2007-01-01 07:10:15.123400')
    assert p.freq == 'us'