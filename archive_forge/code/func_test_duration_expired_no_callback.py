from oslo_messaging._drivers import common
from oslo_messaging import _utils as utils
from oslo_messaging.tests import utils as test_utils
from unittest import mock
@mock.patch('oslo_utils.timeutils.now')
def test_duration_expired_no_callback(self, now):
    now.return_value = 0
    t = common.DecayingTimer(2)
    t.start()
    now.return_value = 3
    remaining = t.check_return()
    self.assertEqual(0, remaining)