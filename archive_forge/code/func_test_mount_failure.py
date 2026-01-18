import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch.object(priv_rootwrap, 'execute')
def test_mount_failure(self, mock_execute):
    err_msg = 'mount.nfs: nfs broke'
    mock_execute.side_effect = putils.ProcessExecutionError(stderr=err_msg)
    client = remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/var/asdf')
    self.assertRaises(putils.ProcessExecutionError, client._do_mount, 'nfs', '192.0.2.20:/share', '/var/asdf')