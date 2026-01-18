from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils.BaseUtils, '_get_wmi_obj')
@mock.patch.object(baseutils, 'sys')
def test_get_wmi_conn_cached(self, mock_sys, mock_get_wmi_obj):
    mock_sys.platform = 'win32'
    baseutils.BaseUtils._WMI_CONS[mock.sentinel.moniker] = mock.sentinel.conn
    result = self.utils._get_wmi_conn(mock.sentinel.moniker)
    self.assertEqual(mock.sentinel.conn, result)
    self.assertFalse(mock_get_wmi_obj.called)