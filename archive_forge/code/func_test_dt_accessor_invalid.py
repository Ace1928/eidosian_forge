import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
import pytz
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas.errors import SettingWithCopyError
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import (
@pytest.mark.parametrize('ser', [Series(np.arange(5)), Series(list('abcde')), Series(np.random.default_rng(2).standard_normal(5))])
def test_dt_accessor_invalid(self, ser):
    with pytest.raises(AttributeError, match='only use .dt accessor'):
        ser.dt
    assert not hasattr(ser, 'dt')