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
@mock.patch.object(utils.priv_nvme, 'create_hostnqn')
@mock.patch.object(builtins, 'open')
def test_get_host_nqn_err(self, mock_open, mock_create):
    mock_open.side_effect = Exception()
    result = utils.get_host_nqn()
    mock_open.assert_called_once_with('/etc/nvme/hostnqn', 'r')
    mock_create.assert_not_called()
    self.assertIsNone(result)