from unittest import mock
import ddt
from os_brick import exception
from os_brick.remotefs import windows_remotefs
from os_brick.tests import base
@ddt.data({'share': '//addr/share_name/subdir_a/subdir_b', 'exp_path': 'C:\\shared_dir\\subdir_a\\subdir_b'}, {'share': '//addr/share_name', 'exp_path': 'C:\\shared_dir'})
@ddt.unpack
@mock.patch('os.path.join', lambda *args: '\\'.join(args))
def test_get_local_share_path(self, share, exp_path):
    fake_local_path = 'C:\\shared_dir'
    self._smbutils.get_smb_share_path.return_value = fake_local_path
    share_path = self._remotefs.get_local_share_path(share)
    self.assertEqual(exp_path, share_path)
    self._smbutils.get_smb_share_path.assert_called_once_with('share_name')