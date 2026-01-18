import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
def test_init_sets_mount_base(self):
    client = remotefs.RemoteFsClient('cifs', root_helper='true', smbfs_mount_point_base='/fake', cifs_mount_point_base='/fake2')
    self.assertEqual('/fake', client._mount_base)