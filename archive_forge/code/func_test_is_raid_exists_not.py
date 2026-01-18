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
@mock.patch.object(executor.Executor, '_execute')
def test_is_raid_exists_not(self, mock_execute):
    mock_execute.return_value = (VOL_UUID + '\n', '')
    result = self.connector.is_raid_exists(NVME_DEVICE_PATH)
    self.assertEqual(False, result)
    cmd = ['mdadm', '--detail', NVME_DEVICE_PATH]
    args, kwargs = mock_execute.call_args
    self.assertEqual(args[0], cmd[0])
    self.assertEqual(args[1], cmd[1])
    self.assertEqual(args[2], cmd[2])