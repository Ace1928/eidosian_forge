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
@mock.patch.object(nvmeof.NVMeOFConnector, 'get_volume_paths')
@mock.patch.object(nvmeof.NVMeOFConnector, 'run_mdadm')
@mock.patch('os_brick.utils.get_device_size')
def test_extend_volume_replicated(self, mock_device_size, mock_mdadm, mock_paths):
    device_path = '/dev/md/' + connection_properties['alias']
    mock_paths.return_value = [device_path]
    mock_device_size.return_value = 100
    self.assertEqual(100, self.connector.extend_volume(connection_properties))
    mock_paths.assert_called_once_with(mock.ANY)
    self.assertIsInstance(mock_paths.call_args[0][0], nvmeof.NVMeOFConnProps)
    mock_mdadm.assert_called_with(('mdadm', '--grow', '--size', 'max', device_path))
    mock_device_size.assert_called_with(self.connector, device_path)