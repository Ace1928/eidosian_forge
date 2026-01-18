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
@mock.patch.object(nvmeof, 'sysfs_property', return_value='live')
def test_ctrl_property(self, mock_sysfs):
    """Controller properties just read from nvme fabrics in sysfs."""
    res = nvmeof.ctrl_property('state', 'nvme0')
    self.assertEqual('live', res)
    mock_sysfs.assert_called_once_with('state', '/sys/class/nvme-fabrics/ctl/nvme0')