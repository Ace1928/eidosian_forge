from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_new_disk_policy(self):
    mock_setting_obj = mock.Mock()
    setting_cls = self._diskutils._conn_storage.MSFT_StorageSetting
    setting_cls.Get.return_value = (0, mock_setting_obj)
    policy = self._diskutils.get_new_disk_policy()
    self.assertEqual(mock_setting_obj.NewDiskPolicy, policy)