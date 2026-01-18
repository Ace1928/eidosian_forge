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
@ddt.data(True, False)
@mock.patch.object(nvmeof.NVMeOFConnector, 'native_multipath_supported', None)
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_native_multipath_supported')
def test__set_native_multipath_supported(self, value, mock_ana):
    mock_ana.return_value = value
    res = self.connector._set_native_multipath_supported()
    mock_ana.assert_called_once_with()
    self.assertIs(value, res)