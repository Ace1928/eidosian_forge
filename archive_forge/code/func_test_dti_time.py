import calendar
from datetime import (
import locale
import unicodedata
import numpy as np
import pytest
from pandas._libs.tslibs import timezones
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import DatetimeArray
def test_dti_time(self):
    rng = date_range('1/1/2000', freq='12min', periods=10)
    result = Index(rng).time
    expected = [t.time() for t in rng]
    assert (result == expected).all()