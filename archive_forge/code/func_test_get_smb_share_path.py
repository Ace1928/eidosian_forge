from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_get_smb_share_path(self):
    fake_share = mock.Mock(Path=mock.sentinel.share_path)
    self._smb_conn.Msft_SmbShare.return_value = [fake_share]
    share_path = self._smbutils.get_smb_share_path(mock.sentinel.share_name)
    self.assertEqual(mock.sentinel.share_path, share_path)
    self._smb_conn.Msft_SmbShare.assert_called_once_with(Name=mock.sentinel.share_name)