from datetime import (
from hypothesis import given
import numpy as np
import pytest
from pandas._libs.tslibs import ccalendar
from pandas._testing._hypothesis import DATETIME_IN_PD_TIMESTAMP_RANGE_NO_TZ
@pytest.mark.parametrize('input_date_tuple, expected_iso_tuple', [[(2020, 1, 1), (2020, 1, 3)], [(2019, 12, 31), (2020, 1, 2)], [(2019, 12, 30), (2020, 1, 1)], [(2009, 12, 31), (2009, 53, 4)], [(2010, 1, 1), (2009, 53, 5)], [(2010, 1, 3), (2009, 53, 7)], [(2010, 1, 4), (2010, 1, 1)], [(2006, 1, 1), (2005, 52, 7)], [(2005, 12, 31), (2005, 52, 6)], [(2008, 12, 28), (2008, 52, 7)], [(2008, 12, 29), (2009, 1, 1)]])
def test_dt_correct_iso_8601_year_week_and_day(input_date_tuple, expected_iso_tuple):
    result = ccalendar.get_iso_calendar(*input_date_tuple)
    expected_from_date_isocalendar = date(*input_date_tuple).isocalendar()
    assert result == expected_from_date_isocalendar
    assert result == expected_iso_tuple