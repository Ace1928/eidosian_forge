from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
@ddt.data(True, False)
@mock.patch.object(networkutils.NetworkUtils, '_get_setting_data')
def test_get_switch_port_allocation(self, enable_cache, mock_get_set_data):
    self.netutils._enable_cache = enable_cache
    self.netutils._switch_ports = {}
    mock_get_set_data.return_value = (mock.sentinel.port, True)
    port, found = self.netutils._get_switch_port_allocation(mock.sentinel.port_name)
    self.assertEqual(mock.sentinel.port, port)
    self.assertTrue(found)
    expected_cache = {mock.sentinel.port_name: port} if enable_cache else {}
    self.assertEqual(expected_cache, self.netutils._switch_ports)
    mock_get_set_data.assert_called_once_with(self.netutils._PORT_ALLOC_SET_DATA, mock.sentinel.port_name, False)