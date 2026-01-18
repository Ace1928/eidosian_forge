from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_vswitch_not_found(self):
    self.netutils._switches = {}
    self.netutils._conn.Msvm_VirtualEthernetSwitch.return_value = []
    self.assertRaises(exceptions.HyperVvSwitchNotFound, self.netutils._get_vswitch, self._FAKE_VSWITCH_NAME)