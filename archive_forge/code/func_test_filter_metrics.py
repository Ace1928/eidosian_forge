from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_filter_metrics(self):
    mock_metric = mock.MagicMock(MetricDefinitionId=mock.sentinel.def_id)
    mock_bad_metric = mock.MagicMock()
    mock_metric_def = mock.MagicMock(Id=mock.sentinel.def_id)
    result = self.utils._filter_metrics([mock_bad_metric, mock_metric], mock_metric_def)
    self.assertEqual([mock_metric], result)