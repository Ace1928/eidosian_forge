import builtins
import errno
import os.path
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick import executor
from os_brick.initiator.connectors import nvmeof
from os_brick.privileged import nvmeof as priv_nvmeof
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base as test_base
from os_brick.tests.initiator import test_connector
from os_brick import utils
@mock.patch('glob.glob', return_value=['/dev/nvme0n1', '/dev/nvme1n1'])
def test__get_nvme_devices(self, mock_glob):
    """Test getting all nvme devices present in system."""
    res = self.target._get_nvme_devices()
    self.assertEqual(mock_glob.return_value, res)
    mock_glob.assert_called_once_with('/dev/nvme*n*')