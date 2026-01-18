from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_vm_has_s3_controller')
@mock.patch.object(vmutils.VMUtils, '_get_new_resource_setting_data')
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_disable_remotefx_video_adapter(self, mock_get_element_associated_class, mock_get_new_rsd, mock_vm_has_s3_controller):
    mock_vm = self._lookup_vm()
    mock_r1 = mock.MagicMock(ResourceSubType=self._vmutils._REMOTEFX_DISP_CTRL_RES_SUB_TYPE)
    mock_r2 = mock.MagicMock(ResourceSubType=self._vmutils._S3_DISP_CTRL_RES_SUB_TYPE)
    mock_get_element_associated_class.return_value = [mock_r1, mock_r2]
    self._vmutils.disable_remotefx_video_adapter(mock.sentinel.fake_vm_name)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_instance_id=mock_vm.InstanceID)
    self._vmutils._jobutils.remove_virt_resource.assert_called_once_with(mock_r1)
    mock_get_new_rsd.assert_called_once_with(self._vmutils._SYNTH_DISP_CTRL_RES_SUB_TYPE, self._vmutils._SYNTH_DISP_ALLOCATION_SETTING_DATA_CLASS)
    self._vmutils._jobutils.add_virt_resource.assert_called_once_with(mock_get_new_rsd.return_value, mock_vm)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_r2)
    self.assertEqual(self._vmutils._DISP_CTRL_ADDRESS, mock_r2.Address)