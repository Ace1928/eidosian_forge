from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(_wqlutils, 'get_element_associated_class')
@mock.patch.object(metricsutils.MetricsUtils, '_sum_metrics_values_by_defs')
@mock.patch.object(metricsutils.MetricsUtils, '_get_metrics_value_instances')
@mock.patch.object(metricsutils.MetricsUtils, '_get_vm_resources')
def test_get_vnic_metrics(self, mock_get_vm_resources, mock_get_value_instances, mock_sum_by_defs, mock_get_element_associated_class):
    fake_rx_mb = 1000
    fake_tx_mb = 2000
    self.utils._metrics_defs_obj = {self.utils._NET_IN_METRICS: mock.sentinel.net_in_metrics, self.utils._NET_OUT_METRICS: mock.sentinel.net_out_metrics}
    mock_port = mock.MagicMock(Parent=mock.sentinel.vnic_path)
    mock_vnic = mock.MagicMock(ElementName=mock.sentinel.element_name, Address=mock.sentinel.address)
    mock_vnic.path_.return_value = mock.sentinel.vnic_path
    mock_get_vm_resources.side_effect = [[mock_port], [mock_vnic]]
    mock_sum_by_defs.return_value = [fake_rx_mb, fake_tx_mb]
    vnic_metrics = list(self.utils.get_vnic_metrics(mock.sentinel.vm_name))
    self.assertEqual(1, len(vnic_metrics))
    self.assertEqual(fake_rx_mb, vnic_metrics[0]['rx_mb'])
    self.assertEqual(fake_tx_mb, vnic_metrics[0]['tx_mb'])
    self.assertEqual(mock.sentinel.element_name, vnic_metrics[0]['element_name'])
    self.assertEqual(mock.sentinel.address, vnic_metrics[0]['address'])
    mock_get_vm_resources.assert_has_calls([mock.call(mock.sentinel.vm_name, self.utils._PORT_ALLOC_SET_DATA), mock.call(mock.sentinel.vm_name, self.utils._SYNTH_ETH_PORT_SET_DATA)])
    mock_get_value_instances.assert_called_once_with(mock_get_element_associated_class.return_value, self.utils._BASE_METRICS_VALUE)
    mock_sum_by_defs.assert_called_once_with(mock_get_value_instances.return_value, [mock.sentinel.net_in_metrics, mock.sentinel.net_out_metrics])