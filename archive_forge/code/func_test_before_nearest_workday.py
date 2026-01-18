from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_THURSDAY, _WEDNESDAY), (_FRIDAY, _THURSDAY), (_SATURDAY, _THURSDAY), (_SUNDAY, _FRIDAY), (_MONDAY, _FRIDAY), (_TUESDAY, _MONDAY), (_NEXT_WEDNESDAY, _TUESDAY)])
def test_before_nearest_workday(day, expected):
    assert before_nearest_workday(day) == expected