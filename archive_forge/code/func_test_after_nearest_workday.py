from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_SATURDAY, _MONDAY), (_SUNDAY, _TUESDAY), (_FRIDAY, _MONDAY)])
def test_after_nearest_workday(day, expected):
    assert after_nearest_workday(day) == expected