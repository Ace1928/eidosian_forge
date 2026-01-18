from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_unexisting_disk_by_unique_id(self):
    mock_msft_disk_cls = self._diskutils._conn_storage.Msft_Disk
    mock_msft_disk_cls.return_value = []
    self.assertRaises(exceptions.DiskNotFound, self._diskutils._get_disks_by_unique_id, mock.sentinel.unique_id, mock.sentinel.unique_id_format)