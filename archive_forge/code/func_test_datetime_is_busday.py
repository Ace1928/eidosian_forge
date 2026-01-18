import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_is_busday(self):
    holidays = ['2011-01-01', '2011-10-10', '2011-11-11', '2011-11-24', '2011-12-25', '2011-05-30', '2011-02-21', '2011-01-17', '2011-12-26', '2012-01-02', '2011-02-21', '2011-05-30', '2011-07-01', '2011-07-04', '2011-09-05', '2011-10-10', 'NaT']
    bdd = np.busdaycalendar(weekmask='1111100', holidays=holidays)
    assert_equal(np.is_busday('2011-01-01'), False)
    assert_equal(np.is_busday('2011-01-02'), False)
    assert_equal(np.is_busday('2011-01-03'), True)
    assert_equal(np.is_busday(holidays, busdaycal=bdd), np.zeros(len(holidays), dtype='?'))