from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
def test_get_wmi_conn_kwargs(self):
    self.utils._WMI_CONS.clear()
    self._check_get_wmi_conn(privileges=mock.sentinel.privileges)
    self.assertNotIn(mock.sentinel.moniker, baseutils.BaseUtils._WMI_CONS)