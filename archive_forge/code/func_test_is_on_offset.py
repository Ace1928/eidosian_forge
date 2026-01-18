from __future__ import annotations
from datetime import datetime
import pytest
import pandas._testing as tm
from pandas.tests.tseries.offsets.common import (
from pandas.tseries.offsets import (
@pytest.mark.parametrize('case', on_offset_cases)
def test_is_on_offset(self, case):
    offset, dt, expected = case
    assert_is_on_offset(offset, dt, expected)