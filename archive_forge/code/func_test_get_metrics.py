from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_filter_metrics')
def test_get_metrics(self, mock_filter_metrics):
    mock_metric = mock.MagicMock()
    mock_element = mock.MagicMock()
    self.utils._conn.Msvm_MetricForME.return_value = [mock_metric]
    result = self.utils._get_metrics(mock_element, mock.sentinel.metrics_def)
    self.assertEqual(mock_filter_metrics.return_value, result)
    self.utils._conn.Msvm_MetricForME.assert_called_once_with(Antecedent=mock_element.path_.return_value)
    mock_filter_metrics.assert_called_once_with([mock_metric.Dependent], mock.sentinel.metrics_def)