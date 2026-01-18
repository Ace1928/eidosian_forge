import os
from unittest import mock
import ddt
from os_brick.initiator.windows import smbfs
from os_brick.remotefs import windows_remotefs
from os_brick.tests.windows import test_base
@mock.patch.object(smbfs.WindowsSMBFSConnector, '_get_export_path')
def test_ensure_share_mounted(self, mock_get_export_path):
    fake_conn_props = dict(options=mock.sentinel.mount_opts)
    self._connector.ensure_share_mounted(fake_conn_props)
    self._remotefs.mount.assert_called_once_with(mock_get_export_path.return_value, mock.sentinel.mount_opts)