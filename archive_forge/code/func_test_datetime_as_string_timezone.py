import numpy
import numpy as np
import datetime
import pytest
from numpy.testing import (
from numpy.compat import pickle
@pytest.mark.skipif(not _has_pytz, reason='The pytz module is not available.')
def test_datetime_as_string_timezone(self):
    a = np.datetime64('2010-03-15T06:30', 'm')
    assert_equal(np.datetime_as_string(a), '2010-03-15T06:30')
    assert_equal(np.datetime_as_string(a, timezone='naive'), '2010-03-15T06:30')
    assert_equal(np.datetime_as_string(a, timezone='UTC'), '2010-03-15T06:30Z')
    assert_(np.datetime_as_string(a, timezone='local') != '2010-03-15T06:30')
    b = np.datetime64('2010-02-15T06:30', 'm')
    assert_equal(np.datetime_as_string(a, timezone=tz('US/Central')), '2010-03-15T01:30-0500')
    assert_equal(np.datetime_as_string(a, timezone=tz('US/Eastern')), '2010-03-15T02:30-0400')
    assert_equal(np.datetime_as_string(a, timezone=tz('US/Pacific')), '2010-03-14T23:30-0700')
    assert_equal(np.datetime_as_string(b, timezone=tz('US/Central')), '2010-02-15T00:30-0600')
    assert_equal(np.datetime_as_string(b, timezone=tz('US/Eastern')), '2010-02-15T01:30-0500')
    assert_equal(np.datetime_as_string(b, timezone=tz('US/Pacific')), '2010-02-14T22:30-0800')
    assert_raises(TypeError, np.datetime_as_string, a, unit='D', timezone=tz('US/Pacific'))
    assert_equal(np.datetime_as_string(a, unit='D', timezone=tz('US/Pacific'), casting='unsafe'), '2010-03-14')
    assert_equal(np.datetime_as_string(b, unit='D', timezone=tz('US/Central'), casting='unsafe'), '2010-02-15')