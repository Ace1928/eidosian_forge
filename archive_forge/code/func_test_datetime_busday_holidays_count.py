import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_busday_holidays_count(self):
    holidays = ['2011-01-01', '2011-10-10', '2011-11-11', '2011-11-24', '2011-12-25', '2011-05-30', '2011-02-21', '2011-01-17', '2011-12-26', '2012-01-02', '2011-02-21', '2011-05-30', '2011-07-01', '2011-07-04', '2011-09-05', '2011-10-10']
    bdd = np.busdaycalendar(weekmask='1111100', holidays=holidays)
    dates = np.busday_offset('2011-01-01', np.arange(366), roll='forward', busdaycal=bdd)
    assert_equal(np.busday_count('2011-01-01', dates, busdaycal=bdd), np.arange(366))
    assert_equal(np.busday_count(dates, '2011-01-01', busdaycal=bdd), -np.arange(366) - 1)
    dates = np.busday_offset('2011-12-31', -np.arange(366), roll='forward', busdaycal=bdd)
    expected = np.arange(366)
    expected[0] = -1
    assert_equal(np.busday_count(dates, '2011-12-31', busdaycal=bdd), expected)
    expected = -np.arange(366) + 1
    expected[0] = 0
    assert_equal(np.busday_count('2011-12-31', dates, busdaycal=bdd), expected)
    assert_raises(ValueError, np.busday_offset, '2012-01-03', '2012-02-03', weekmask='1111100', busdaycal=bdd)
    assert_raises(ValueError, np.busday_offset, '2012-01-03', '2012-02-03', holidays=holidays, busdaycal=bdd)
    assert_equal(np.busday_count('2011-03', '2011-04', weekmask='Mon'), 4)
    assert_equal(np.busday_count('2011-04', '2011-03', weekmask='Mon'), -4)
    sunday = np.datetime64('2023-03-05')
    monday = sunday + 1
    friday = sunday + 5
    saturday = sunday + 6
    assert_equal(np.busday_count(sunday, monday), 0)
    assert_equal(np.busday_count(monday, sunday), -1)
    assert_equal(np.busday_count(friday, saturday), 1)
    assert_equal(np.busday_count(saturday, friday), 0)