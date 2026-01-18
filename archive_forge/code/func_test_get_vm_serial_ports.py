from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_serial_ports(self, mock_get_element_associated_class):
    mock_vmsettings = self._lookup_vm()
    fake_serial_port = mock.MagicMock()
    fake_serial_port.ResourceSubType = self._vmutils._SERIAL_PORT_RES_SUB_TYPE
    mock_rasds = [fake_serial_port]
    mock_get_element_associated_class.return_value = mock_rasds
    ret_val = self._vmutils._get_vm_serial_ports(mock_vmsettings)
    self.assertEqual(mock_rasds, ret_val)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._SERIAL_PORT_SETTING_DATA_CLASS, element_instance_id=mock_vmsettings.InstanceID)