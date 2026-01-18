from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(True, False)
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
def test_create_vm(self, mock_get_wmi_obj, vnuma_enabled=True):
    mock_vs_man_svc = self._vmutils._vs_man_svc
    mock_vs_data = mock.MagicMock()
    fake_job_path = 'fake job path'
    fake_ret_val = 'fake return value'
    fake_vm_name = 'fake_vm_name'
    _conn = self._vmutils._conn.Msvm_VirtualSystemSettingData
    self._vmutils._jobutils.check_ret_val.return_value = mock.sentinel.job
    _conn.new.return_value = mock_vs_data
    mock_vs_man_svc.DefineSystem.return_value = (fake_job_path, mock.sentinel.vm_path, fake_ret_val)
    self._vmutils.create_vm(vm_name=fake_vm_name, vm_gen=constants.VM_GEN_2, notes='fake notes', vnuma_enabled=vnuma_enabled, instance_path=mock.sentinel.instance_path)
    _conn.new.assert_called_once_with()
    self.assertEqual(mock_vs_data.ElementName, fake_vm_name)
    mock_vs_man_svc.DefineSystem.assert_called_once_with(ResourceSettings=[], ReferenceConfiguration=None, SystemSettings=mock_vs_data.GetText_(1))
    self._vmutils._jobutils.check_ret_val.assert_called_once_with(fake_ret_val, fake_job_path)
    self.assertEqual(self._vmutils._VIRTUAL_SYSTEM_SUBTYPE_GEN2, mock_vs_data.VirtualSystemSubType)
    self.assertFalse(mock_vs_data.SecureBootEnabled)
    self.assertEqual(vnuma_enabled, mock_vs_data.VirtualNumaEnabled)
    self.assertEqual(self._vmutils._VIRTUAL_SYSTEM_SUBTYPE_GEN2, mock_vs_data.VirtualSystemSubType)
    self.assertEqual(mock_vs_data.Notes, 'fake notes')
    self.assertEqual(mock.sentinel.instance_path, mock_vs_data.ConfigurationDataRoot)
    self.assertEqual(mock.sentinel.instance_path, mock_vs_data.LogDataRoot)
    self.assertEqual(mock.sentinel.instance_path, mock_vs_data.SnapshotDataRoot)
    self.assertEqual(mock.sentinel.instance_path, mock_vs_data.SuspendDataRoot)
    self.assertEqual(mock.sentinel.instance_path, mock_vs_data.SwapFileDataRoot)