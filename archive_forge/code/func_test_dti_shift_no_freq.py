from datetime import datetime
import pytest
import pytz
from pandas.errors import NullFrequencyError
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_dti_shift_no_freq(self, unit):
    dti = DatetimeIndex(['2011-01-01 10:00', '2011-01-01'], freq=None).as_unit(unit)
    with pytest.raises(NullFrequencyError, match='Cannot shift with no freq'):
        dti.shift(2)