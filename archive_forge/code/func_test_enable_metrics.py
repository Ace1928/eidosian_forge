from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
def test_enable_metrics(self):
    metrics_name = self.utils._CPU_METRICS
    metrics_def = mock.MagicMock()
    self.utils._metrics_defs_obj = {metrics_name: metrics_def}
    self._check_enable_metrics([metrics_name, mock.sentinel.metrics_name], metrics_def.path_.return_value)