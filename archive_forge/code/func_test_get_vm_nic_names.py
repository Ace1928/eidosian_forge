from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_vm_nics')
def test_get_vm_nic_names(self, mock_get_vm_nics):
    exp_nic_names = ['port1', 'port2']
    mock_get_vm_nics.return_value = [mock.Mock(ElementName=nic_name) for nic_name in exp_nic_names]
    nic_names = self._vmutils.get_vm_nic_names(mock.sentinel.vm_name)
    self.assertEqual(exp_nic_names, nic_names)
    mock_get_vm_nics.assert_called_once_with(mock.sentinel.vm_name)