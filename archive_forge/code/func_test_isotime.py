import calendar
import datetime
from unittest import mock
import iso8601
from glance.common import timeutils
from glance.tests import utils as test_utils
def test_isotime(self):
    with mock.patch('datetime.datetime') as datetime_mock:
        datetime_mock.utcnow.return_value = self.skynet_self_aware_time
        dt = timeutils.isotime()
        self.assertEqual(dt, self.skynet_self_aware_time_str)