from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_SATURDAY, _FRIDAY), (_SUNDAY, _MONDAY), (_MONDAY, _MONDAY)])
def test_nearest_workday(day, expected):
    assert nearest_workday(day) == expected