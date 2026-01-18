import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_isotimei_micro_second_precision(self):
    with mock.patch('datetime.datetime') as datetime_mock:
        datetime_mock.utcnow.return_value = self.skynet_self_aware_ms_time
        dt = timeutils.isotime(subsecond=True)
        self.assertEqual(dt, self.skynet_self_aware_time_ms_str)