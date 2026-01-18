from unittest import mock
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
from os_win.utils.metrics import metricsutils
from os_win import utilsfactory
@mock.patch.object(metricsutils.MetricsUtils, '_unique_result')
def test_get_switch_port(self, mock_unique_result):
    result = self.utils._get_switch_port(mock.sentinel.port_name)
    self.assertEqual(mock_unique_result.return_value, result)
    conn_class = self.utils._conn.Msvm_EthernetPortAllocationSettingData
    conn_class.assert_called_once_with(ElementName=mock.sentinel.port_name)
    mock_unique_result.assert_called_once_with(conn_class.return_value, mock.sentinel.port_name)