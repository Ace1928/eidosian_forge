from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_vswitch_cache(self):
    self.netutils._switches = {self._FAKE_VSWITCH_NAME: mock.sentinel.vswitch}
    vswitch = self.netutils._get_vswitch(self._FAKE_VSWITCH_NAME)
    self.assertEqual(mock.sentinel.vswitch, vswitch)