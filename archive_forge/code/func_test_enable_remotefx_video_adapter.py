from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(vmutils.VMUtils, '_set_remotefx_display_controller')
@mock.patch.object(vmutils.VMUtils, '_vm_has_s3_controller')
def test_enable_remotefx_video_adapter(self, mock_vm_has_s3_controller, mock_set_remotefx_ctrl, mock_get_element_associated_class):
    mock_vm = self._lookup_vm()
    mock_r1 = mock.MagicMock()
    mock_r1.ResourceSubType = self._vmutils._SYNTH_DISP_CTRL_RES_SUB_TYPE
    mock_r2 = mock.MagicMock()
    mock_r2.ResourceSubType = self._vmutils._S3_DISP_CTRL_RES_SUB_TYPE
    mock_get_element_associated_class.return_value = [mock_r1, mock_r2]
    self._vmutils.enable_remotefx_video_adapter(mock.sentinel.fake_vm_name, self._FAKE_MONITOR_COUNT, constants.REMOTEFX_MAX_RES_1024x768)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._conn, self._vmutils._CIM_RES_ALLOC_SETTING_DATA_CLASS, element_instance_id=mock_vm.InstanceID)
    self._vmutils._jobutils.remove_virt_resource.assert_called_once_with(mock_r1)
    mock_set_remotefx_ctrl.assert_called_once_with(mock_vm, None, self._FAKE_MONITOR_COUNT, self._vmutils._remote_fx_res_map[constants.REMOTEFX_MAX_RES_1024x768], None)
    self._vmutils._jobutils.modify_virt_resource.assert_called_once_with(mock_r2)
    self.assertEqual(self._vmutils._DISP_CTRL_ADDRESS_DX_11, mock_r2.Address)