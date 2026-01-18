from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_conn')
def test_conn(self, mock_get_wmi_conn):
    self.utils._conn_attr = None
    self.assertEqual(mock_get_wmi_conn.return_value, self.utils._conn)
    mock_get_wmi_conn.assert_called_once_with(self.utils._wmi_namespace % '.')