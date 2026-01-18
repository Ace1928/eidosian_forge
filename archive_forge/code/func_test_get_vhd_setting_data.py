import platform
from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import livemigrationutils
from os_win.utils.compute import vmutils
from os_win.utils import jobutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vhd_setting_data(self, mock_get_elem_associated_class):
    self._prepare_vm_mocks(self._RESOURCE_TYPE_VHD, self._RESOURCE_SUB_TYPE_VHD, mock_get_elem_associated_class)
    mock_vm = mock.Mock(Name='fake_vm_name')
    mock_sasd = mock_get_elem_associated_class.return_value[0]
    vhd_sds = self.liveutils._get_vhd_setting_data(mock_vm)
    self.assertEqual([mock_sasd.GetText_.return_value], vhd_sds)
    mock_get_elem_associated_class.assert_called_once_with(self._conn, self.liveutils._STORAGE_ALLOC_SETTING_DATA_CLASS, element_uuid=mock_vm.Name)