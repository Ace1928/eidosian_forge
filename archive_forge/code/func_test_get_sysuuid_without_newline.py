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
@mock.patch.object(nvmeof.NVMeOFConnector, '_execute', autospec=True)
def test_get_sysuuid_without_newline(self, mock_execute):
    mock_execute.return_value = ('9126E942-396D-11E7-B0B7-A81E84C186D1\n', '')
    uuid = self.connector._get_host_uuid()
    expected_uuid = '9126E942-396D-11E7-B0B7-A81E84C186D1'
    self.assertEqual(expected_uuid, uuid)