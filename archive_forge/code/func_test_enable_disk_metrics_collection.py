from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_enable_metrics')
def test_enable_disk_metrics_collection(self, mock_enable_metrics):
    mock_get_disk = self.utils._vmutils._get_mounted_disk_resource_from_path
    self.utils.enable_disk_metrics_collection(mock.sentinel.disk_path, mock.sentinel.is_physical, mock.sentinel.serial)
    mock_get_disk.assert_called_once_with(mock.sentinel.disk_path, is_physical=mock.sentinel.is_physical, serial=mock.sentinel.serial)
    mock_enable_metrics.assert_called_once_with(mock_get_disk.return_value)