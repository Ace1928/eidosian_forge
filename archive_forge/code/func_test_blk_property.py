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
@mock.patch.object(nvmeof, 'sysfs_property', return_value='uuid_value')
def test_blk_property(self, mock_sysfs):
    """Block properties just read from block devices in sysfs."""
    res = nvmeof.blk_property('uuid', 'nvme0n1')
    self.assertEqual('uuid_value', res)
    mock_sysfs.assert_called_once_with('uuid', '/sys/class/block/nvme0n1')