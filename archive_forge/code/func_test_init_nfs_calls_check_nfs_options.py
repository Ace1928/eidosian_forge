import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
@mock.patch('os_brick.remotefs.remotefs.RemoteFsClient._check_nfs_options')
def test_init_nfs_calls_check_nfs_options(self, mock_check_nfs_options):
    remotefs.RemoteFsClient('nfs', root_helper='true', nfs_mount_point_base='/fake')
    mock_check_nfs_options.assert_called_once_with()