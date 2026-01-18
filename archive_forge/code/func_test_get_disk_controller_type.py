from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@ddt.data(vmutils.VMUtils._SCSI_CTRL_RES_SUB_TYPE, vmutils.VMUtils._IDE_CTRL_RES_SUB_TYPE)
@mock.patch.object(vmutils.VMUtils, '_get_wmi_obj')
def test_get_disk_controller_type(self, res_sub_type, mock_get_wmi_obj):
    mock_ctrl = mock_get_wmi_obj.return_value
    mock_ctrl.ResourceSubType = res_sub_type
    exp_ctrl_type = self._vmutils._disk_ctrl_type_mapping[res_sub_type]
    ctrl_type = self._vmutils._get_disk_controller_type(mock.sentinel.ctrl_path)
    self.assertEqual(exp_ctrl_type, ctrl_type)
    mock_get_wmi_obj.assert_called_once_with(mock.sentinel.ctrl_path)