from unittest import mock
import six
import importlib
from os_win.tests.unit import test_base
from os_win.utils import baseutils
@mock.patch.object(baseutils.BaseUtilsVirt, '_get_wmi_compat_conn')
def test_get_wmi_obj_compatibility_6_2(self, mock_get_wmi_compat):
    baseutils.BaseUtilsVirt._os_version = [6, 2]
    result = self.utils._get_wmi_obj(mock.sentinel.moniker, True)
    self.assertEqual(mock_get_wmi_compat.return_value, result)