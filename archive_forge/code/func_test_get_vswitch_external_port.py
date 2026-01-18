from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_vswitch_external_port(self):
    vswitch = mock.MagicMock(Name=mock.sentinel.vswitch_name)
    self.netutils._conn.Msvm_VirtualEthernetSwitch.return_value = [vswitch]
    conn = self.netutils._conn
    ext_port = mock.MagicMock()
    lan_endpoint_assoc1 = mock.MagicMock()
    lan_endpoint_assoc2 = mock.Mock(SystemName=mock.sentinel.vswitch_name)
    self.netutils._conn.Msvm_ExternalEthernetPort.return_value = [ext_port]
    conn.Msvm_EthernetDeviceSAPImplementation.return_value = [lan_endpoint_assoc1]
    conn.Msvm_ActiveConnection.return_value = [mock.Mock(Antecedent=lan_endpoint_assoc2)]
    result = self.netutils._get_vswitch_external_port(mock.sentinel.name)
    self.assertEqual(ext_port, result)
    conn.Msvm_EthernetDeviceSAPImplementation.assert_called_once_with(Antecedent=ext_port.path_.return_value)
    conn.Msvm_ActiveConnection.assert_called_once_with(Dependent=lan_endpoint_assoc1.Dependent.path_.return_value)