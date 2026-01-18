from datetime import datetime
import pprint
import dateutil.tz
import pytest
import pytz  # a test below uses pytz but only inside a `eval` call
from pandas import Timestamp
@pytest.mark.parametrize('ts, timespec, expected_iso', [(ts_no_ns, 'auto', '2019-05-18T15:17:08.132263'), (ts_no_ns, 'seconds', '2019-05-18T15:17:08'), (ts_no_ns, 'nanoseconds', '2019-05-18T15:17:08.132263000'), (ts_no_ns_year1, 'seconds', '0001-05-18T15:17:08'), (ts_no_ns_year1, 'nanoseconds', '0001-05-18T15:17:08.132263000'), (ts_ns, 'auto', '2019-05-18T15:17:08.132263123'), (ts_ns, 'hours', '2019-05-18T15'), (ts_ns, 'minutes', '2019-05-18T15:17'), (ts_ns, 'seconds', '2019-05-18T15:17:08'), (ts_ns, 'milliseconds', '2019-05-18T15:17:08.132'), (ts_ns, 'microseconds', '2019-05-18T15:17:08.132263'), (ts_ns, 'nanoseconds', '2019-05-18T15:17:08.132263123'), (ts_ns_tz, 'auto', '2019-05-18T15:17:08.132263123+00:00'), (ts_ns_tz, 'hours', '2019-05-18T15+00:00'), (ts_ns_tz, 'minutes', '2019-05-18T15:17+00:00'), (ts_ns_tz, 'seconds', '2019-05-18T15:17:08+00:00'), (ts_ns_tz, 'milliseconds', '2019-05-18T15:17:08.132+00:00'), (ts_ns_tz, 'microseconds', '2019-05-18T15:17:08.132263+00:00'), (ts_ns_tz, 'nanoseconds', '2019-05-18T15:17:08.132263123+00:00'), (ts_no_us, 'auto', '2019-05-18T15:17:08.000000123')])
def test_isoformat(ts, timespec, expected_iso):
    assert ts.isoformat(timespec=timespec) == expected_iso