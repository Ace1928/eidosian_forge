import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@mock.patch.object(iscsi.LOG, 'warning')
@mock.patch.object(linuxscsi.LinuxSCSI, 'extend_volume')
@mock.patch.object(iscsi.ISCSIConnector, 'get_volume_paths')
def test_extend_volume_mask_password_no_paths(self, mock_volume_paths, mock_scsi_extend, mock_log_warning):
    fake_new_size = 1024
    mock_volume_paths.return_value = []
    mock_scsi_extend.return_value = fake_new_size
    volume = {'id': 'fake_uuid'}
    connection_info = self.iscsi_connection_chap(volume, '10.0.2.15:3260', 'fake_iqn', 'CHAP', 'fake_user', 'fake_password', 'CHAP1', 'fake_user1', 'fake_password1')
    self.assertRaises(exception.VolumePathsNotFound, self.connector.extend_volume, connection_info['data'])
    self.assertEqual(1, mock_log_warning.call_count)
    self.assertIn("'auth_password': '***'", str(mock_log_warning.call_args_list[0]))
    self.assertIn("'discovery_auth_password': '***'", str(mock_log_warning.call_args_list[0]))