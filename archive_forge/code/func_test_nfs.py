import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch.object(remotefs.RemoteFsClient, '_read_mounts', return_value=[])
def test_nfs(self, mock_read_mounts):
    client = remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/mnt')
    share = '10.0.0.1:/qwe'
    mount_point = client.get_mount_point(share)
    client.mount(share)
    calls = [mock.call('mkdir', '-p', mount_point, check_exit_code=0), mock.call('mount', '-t', 'nfs', '-o', 'vers=4,minorversion=1', share, mount_point, check_exit_code=0, run_as_root=True, root_helper='true')]
    self.mock_execute.assert_has_calls(calls)