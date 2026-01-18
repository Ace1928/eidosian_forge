from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_get_metrics_value_instances(self):
    FAKE_CLASS_NAME = 'FAKE_CLASS'
    mock_el_metric = mock.MagicMock()
    mock_el_metric_2 = mock.MagicMock()
    mock_el_metric_2.path.return_value = mock.Mock(Class=FAKE_CLASS_NAME)
    self.utils._conn.Msvm_MetricForME.side_effect = [[], [mock.Mock(Dependent=mock_el_metric_2)]]
    returned = self.utils._get_metrics_value_instances([mock_el_metric, mock_el_metric_2], FAKE_CLASS_NAME)
    expected_return = [mock_el_metric_2]
    self.assertEqual(expected_return, returned)