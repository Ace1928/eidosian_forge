import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
def test_datetime_busday_offset(self):
    assert_equal(np.busday_offset('2011-06', 0, roll='forward', weekmask='Mon'), np.datetime64('2011-06-06'))
    assert_equal(np.busday_offset('2011-07', -1, roll='forward', weekmask='Mon'), np.datetime64('2011-06-27'))
    assert_equal(np.busday_offset('2011-07', -1, roll='forward', weekmask='Mon'), np.datetime64('2011-06-27'))
    assert_equal(np.busday_offset('2010-08', 0, roll='backward'), np.datetime64('2010-07-30'))
    assert_equal(np.busday_offset('2010-08', 0, roll='preceding'), np.datetime64('2010-07-30'))
    assert_equal(np.busday_offset('2010-08', 0, roll='modifiedpreceding'), np.datetime64('2010-08-02'))
    assert_equal(np.busday_offset('2010-08', 0, roll='modifiedfollowing'), np.datetime64('2010-08-02'))
    assert_equal(np.busday_offset('2010-08', 0, roll='forward'), np.datetime64('2010-08-02'))
    assert_equal(np.busday_offset('2010-08', 0, roll='following'), np.datetime64('2010-08-02'))
    assert_equal(np.busday_offset('2010-10-30', 0, roll='following'), np.datetime64('2010-11-01'))
    assert_equal(np.busday_offset('2010-10-30', 0, roll='modifiedfollowing'), np.datetime64('2010-10-29'))
    assert_equal(np.busday_offset('2010-10-30', 0, roll='modifiedpreceding'), np.datetime64('2010-10-29'))
    assert_equal(np.busday_offset('2010-10-16', 0, roll='modifiedfollowing'), np.datetime64('2010-10-18'))
    assert_equal(np.busday_offset('2010-10-16', 0, roll='modifiedpreceding'), np.datetime64('2010-10-15'))
    assert_raises(ValueError, np.busday_offset, '2011-06-04', 0)
    assert_equal(np.busday_offset('2006-02-01', 25), np.datetime64('2006-03-08'))
    assert_equal(np.busday_offset('2006-03-08', -25), np.datetime64('2006-02-01'))
    assert_equal(np.busday_offset('2007-02-25', 11, weekmask='SatSun'), np.datetime64('2007-04-07'))
    assert_equal(np.busday_offset('2007-04-07', -11, weekmask='SatSun'), np.datetime64('2007-02-25'))
    assert_equal(np.busday_offset(np.datetime64('NaT'), 1, roll='nat'), np.datetime64('NaT'))
    assert_equal(np.busday_offset(np.datetime64('NaT'), 1, roll='following'), np.datetime64('NaT'))
    assert_equal(np.busday_offset(np.datetime64('NaT'), 1, roll='preceding'), np.datetime64('NaT'))