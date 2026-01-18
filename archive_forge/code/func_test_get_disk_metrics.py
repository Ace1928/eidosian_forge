from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_get_metrics_values')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_resources')
def test_get_disk_metrics(self, mock_get_vm_resources, mock_get_metrics_values):
    fake_read_mb = 1000
    fake_write_mb = 2000
    self.utils._metrics_defs_obj = {self.utils._DISK_RD_METRICS: mock.sentinel.disk_rd_metrics, self.utils._DISK_WR_METRICS: mock.sentinel.disk_wr_metrics}
    mock_disk = mock.MagicMock(HostResource=[mock.sentinel.host_resource], InstanceID=mock.sentinel.instance_id)
    mock_get_vm_resources.return_value = [mock_disk]
    mock_get_metrics_values.return_value = [fake_read_mb, fake_write_mb]
    disk_metrics = list(self.utils.get_disk_metrics(mock.sentinel.vm_name))
    self.assertEqual(1, len(disk_metrics))
    self.assertEqual(fake_read_mb, disk_metrics[0]['read_mb'])
    self.assertEqual(fake_write_mb, disk_metrics[0]['write_mb'])
    self.assertEqual(mock.sentinel.instance_id, disk_metrics[0]['instance_id'])
    self.assertEqual(mock.sentinel.host_resource, disk_metrics[0]['host_resource'])
    mock_get_vm_resources.assert_called_once_with(mock.sentinel.vm_name, self.utils._STORAGE_ALLOC_SETTING_DATA_CLASS)
    metrics = [mock.sentinel.disk_rd_metrics, mock.sentinel.disk_wr_metrics]
    mock_get_metrics_values.assert_called_once_with(mock_disk, metrics)