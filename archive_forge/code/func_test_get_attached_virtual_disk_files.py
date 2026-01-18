from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_attached_virtual_disk_files(self):
    disks = [mock.Mock(), mock.Mock()]
    disk_cls = self._diskutils._conn_storage.Msft_Disk
    disk_cls.return_value = disks
    ret_val = self._diskutils.get_attached_virtual_disk_files()
    exp_ret_val = [dict(location=disk.Location, number=disk.Number, offline=disk.IsOffline, readonly=disk.IsReadOnly) for disk in disks]
    self.assertEqual(exp_ret_val, ret_val)
    disk_cls.assert_called_once_with(BusType=diskutils.BUS_FILE_BACKED_VIRTUAL)