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
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_mdadm')
def test_assemble_raid_simple_err(self, mock_run_mdadm):
    mock_run_mdadm.side_effect = putils.ProcessExecutionError()
    self.assertRaises(putils.ProcessExecutionError, self.connector.assemble_raid, ['/dev/sda'], '/dev/md/md1', True)
    mock_run_mdadm.assert_called_with(['mdadm', '--assemble', '--run', '/dev/md/md1', '-o', '/dev/sda'], True)