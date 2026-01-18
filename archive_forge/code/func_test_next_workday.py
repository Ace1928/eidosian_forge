from datetime import datetime
import pytest
from pandas.tseries.holiday import (
@pytest.mark.parametrize('day,expected', [(_WEDNESDAY, _THURSDAY), (_THURSDAY, _FRIDAY), (_SATURDAY, _MONDAY), (_SUNDAY, _MONDAY), (_MONDAY, _TUESDAY), (_TUESDAY, _NEXT_WEDNESDAY)])
def test_next_workday(day, expected):
    assert next_workday(day) == expected