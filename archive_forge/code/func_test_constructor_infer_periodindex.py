from datetime import (
from decimal import Decimal
import numpy as np
import pytest
from pandas._libs.tslibs.timezones import maybe_get_tz
from pandas import (
import pandas._testing as tm
def test_constructor_infer_periodindex(self):
    xp = period_range('2012-1-1', freq='M', periods=3)
    rs = Index(xp)
    tm.assert_index_equal(rs, xp)
    assert isinstance(rs, PeriodIndex)