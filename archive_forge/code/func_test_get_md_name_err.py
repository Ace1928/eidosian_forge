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
@mock.patch.object(builtins, 'open', side_effect=Exception)
def test_get_md_name_err(self, mock_open):
    result = self.connector.get_md_name(os.path.basename(NVME_NS_PATH))
    self.assertIsNone(result)
    mock_open.assert_called_once_with('/proc/mdstat', 'r')