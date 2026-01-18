from unittest import mock
from oslotest import base
from monascaclient.osc import migration as migr
from monascaclient.v2_0 import notifications
from monascaclient.v2_0 import shell
@mock.patch('monascaclient.osc.migration.make_client')
def test_good_notifications_patch_just_period(self, mc):
    period = 0
    args = '--period ' + str(period)
    data = {'period': period}
    self._patch_test(mc, args, data)