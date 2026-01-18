import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch.object(priv_rootwrap, 'execute')
@mock.patch.object(remotefs.RemoteFsClient, '_do_mount')
def test_mount_already_mounted(self, mock_do_mount, mock_execute):
    share = '10.0.0.1:/share'
    client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/mnt')
    mounts = {client.get_mount_point(share): 'some_dev'}
    with mock.patch.object(client, '_read_mounts', return_value=mounts):
        client.mount(share)
        self.assertEqual(mock_do_mount.call_count, 0)
        self.assertEqual(mock_execute.call_count, 0)