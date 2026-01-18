from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_switch_port_allocation_cached(self):
    self.netutils._switch_ports[mock.sentinel.port_name] = mock.sentinel.port
    port, found = self.netutils._get_switch_port_allocation(mock.sentinel.port_name)
    self.assertEqual(mock.sentinel.port, port)
    self.assertTrue(found)