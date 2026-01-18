import os
import tempfile
from unittest import mock
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.remotefs import remotefs
from os_brick.tests import base
def test_invalid_fs(self):
    self.assertRaises(exception.ProtocolNotSupported, remotefs.RemoteFsClient, 'my_fs', root_helper='true')