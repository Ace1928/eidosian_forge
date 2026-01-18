from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_get_vm_from_res_setting_data(self):
    fake_res_set_instance_id = 'Microsoft:GUID\\SpecificData'
    fake_vm_set_instance_id = 'Microsoft:GUID'
    res_setting_data = mock.Mock(InstanceID=fake_res_set_instance_id)
    conn = self.netutils._conn
    mock_setting_data = conn.Msvm_VirtualSystemSettingData.return_value
    resulted_vm = self.netutils._get_vm_from_res_setting_data(res_setting_data)
    conn.Msvm_VirtualSystemSettingData.assert_called_once_with(InstanceID=fake_vm_set_instance_id)
    conn.Msvm_ComputerSystem.assert_called_once_with(Name=mock_setting_data[0].ConfigurationID)
    expected_result = conn.Msvm_ComputerSystem.return_value[0]
    self.assertEqual(expected_result, resulted_vm)