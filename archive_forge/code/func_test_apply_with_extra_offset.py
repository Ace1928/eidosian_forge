from __future__ import annotations
from datetime import (
import numpy as np
import pytest
from pandas._libs.tslibs.offsets import (
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries import offsets
@pytest.mark.parametrize('case', [(CBMonthEnd(n=1, offset=timedelta(days=5)), {datetime(2021, 3, 1): datetime(2021, 3, 31) + timedelta(days=5), datetime(2021, 4, 17): datetime(2021, 4, 30) + timedelta(days=5)}), (CBMonthEnd(n=2, offset=timedelta(days=40)), {datetime(2021, 3, 10): datetime(2021, 4, 30) + timedelta(days=40), datetime(2021, 4, 30): datetime(2021, 6, 30) + timedelta(days=40)}), (CBMonthEnd(n=1, offset=timedelta(days=-5)), {datetime(2021, 3, 1): datetime(2021, 3, 31) - timedelta(days=5), datetime(2021, 4, 11): datetime(2021, 4, 30) - timedelta(days=5)}), (-2 * CBMonthEnd(n=1, offset=timedelta(days=10)), {datetime(2021, 3, 1): datetime(2021, 1, 29) + timedelta(days=10), datetime(2021, 4, 3): datetime(2021, 2, 26) + timedelta(days=10)}), (CBMonthEnd(n=0, offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 3, 31) + timedelta(days=1), datetime(2021, 4, 1): datetime(2021, 4, 30) + timedelta(days=1)}), (CBMonthEnd(n=1, holidays=['2021-03-31'], offset=timedelta(days=1)), {datetime(2021, 3, 2): datetime(2021, 3, 30) + timedelta(days=1)})])
def test_apply_with_extra_offset(self, case):
    offset, cases = case
    for base, expected in cases.items():
        assert_offset_equal(offset, base, expected)