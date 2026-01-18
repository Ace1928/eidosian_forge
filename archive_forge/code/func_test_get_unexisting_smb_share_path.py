from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_get_unexisting_smb_share_path(self):
    self._smb_conn.Msft_SmbShare.return_value = []
    share_path = self._smbutils.get_smb_share_path(mock.sentinel.share_name)
    self.assertIsNone(share_path)
    self._smb_conn.Msft_SmbShare.assert_called_once_with(Name=mock.sentinel.share_name)