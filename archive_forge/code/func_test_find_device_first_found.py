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
@mock.patch.object(nvmeof.Target, 'get_devices')
def test_find_device_first_found(self, mock_get_devs):
    """Returns the first device found."""
    mock_get_devs.return_value = ['/dev/nvme0n1']
    res = self.target.find_device()
    mock_get_devs.assert_called_once_with(only_live=True, get_one=True)
    self.assertEqual('/dev/nvme0n1', res)