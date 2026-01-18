from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_mount_smb_share(self):
    fake_create = self._smb_conn.Msft_SmbMapping.Create
    self._smbutils.mount_smb_share(mock.sentinel.share_path, mock.sentinel.username, mock.sentinel.password)
    fake_create.assert_called_once_with(RemotePath=mock.sentinel.share_path, UserName=mock.sentinel.username, Password=mock.sentinel.password)