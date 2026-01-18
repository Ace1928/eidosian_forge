from unittest import mock
import ddt
from os_brick import exception
from os_brick.remotefs import windows_remotefs
from os_brick.tests import base
@ddt.data({'is_local_share': False}, {'expect_existing': False})
@ddt.unpack
def test_get_local_share_path_missing(self, expect_existing=True, is_local_share=True):
    self._smbutils.get_smb_share_path.return_value = None
    self._smbutils.is_local_share.return_value = is_local_share
    if expect_existing:
        self.assertRaises(exception.VolumePathsNotFound, self._remotefs.get_local_share_path, self._FAKE_SHARE, expect_existing=expect_existing)
    else:
        share_path = self._remotefs.get_local_share_path(self._FAKE_SHARE, expect_existing=expect_existing)
        self.assertIsNone(share_path)
    self.assertEqual(is_local_share, self._smbutils.get_smb_share_path.called)
    self._smbutils.is_local_share.assert_called_once_with(self._FAKE_SHARE)