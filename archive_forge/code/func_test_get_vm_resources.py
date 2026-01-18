from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_setting_data')
def test_get_vm_resources(self, mock_get_vm_setting_data, mock_get_element_associated_class):
    result = self.utils._get_vm_resources(mock.sentinel.vm_name, mock.sentinel.resource_class)
    mock_get_vm_setting_data.assert_called_once_with(mock.sentinel.vm_name)
    vm_setting_data = mock_get_vm_setting_data.return_value
    mock_get_element_associated_class.assert_called_once_with(self.utils._conn, mock.sentinel.resource_class, element_instance_id=vm_setting_data.InstanceID)
    self.assertEqual(mock_get_element_associated_class.return_value, result)