from unittest import mock
import ddt
import six
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils10
from os_win.utils import jobutils
@mock.patch.object(vmutils10.VMUtils10, '_get_wmi_conn')
def test_conn_msps_no_namespace(self, mock_get_wmi_conn):
    self._vmutils._conn_msps_attr = None
    mock_get_wmi_conn.side_effect = [exceptions.OSWinException]
    self.assertRaises(exceptions.OSWinException, lambda: self._vmutils._conn_msps)
    mock_get_wmi_conn.assert_called_with(self._vmutils._MSPS_NAMESPACE % self._vmutils._host)