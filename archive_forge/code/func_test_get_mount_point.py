import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
def test_get_mount_point(self):
    fsclient = remotefs.ScalityRemoteFsClient('scality', root_helper='true', scality_mount_point_base='/fake')
    self.assertEqual('/fake/path/00', fsclient.get_mount_point('path'))