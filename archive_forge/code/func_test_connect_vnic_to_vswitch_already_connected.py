from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_connect_vnic_to_vswitch_already_connected(self):
    mock_port = self._mock_get_switch_port_alloc()
    mock_port.HostResource = [mock.sentinel.vswitch_path]
    self.netutils.connect_vnic_to_vswitch(mock.sentinel.switch_name, mock.sentinel.port_name)
    self.assertFalse(self.netutils._jobutils.modify_virt_resource.called)