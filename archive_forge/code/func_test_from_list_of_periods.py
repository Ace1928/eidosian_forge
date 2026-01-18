from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_from_list_of_periods(self):
    rng = period_range('1/1/2000', periods=20, freq='D')
    periods = list(rng)
    result = Index(periods)
    assert isinstance(result, PeriodIndex)