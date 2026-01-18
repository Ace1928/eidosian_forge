from unittest import mock
import ddt
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import smbutils
def test_mount_smb_share_failed(self):
    self._smb_conn.Msft_SmbMapping.Create.side_effect = exceptions.x_wmi
    self.assertRaises(exceptions.SMBException, self._smbutils.mount_smb_share, mock.sentinel.share_path)