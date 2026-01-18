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
@mock.patch.object(nvmeof.NVMeOFConnector, 'rescan')
@mock.patch.object(nvmeof, 'blk_property')
@mock.patch.object(nvmeof.NVMeOFConnector, '_get_sizes_from_lba')
@mock.patch.object(nvmeof.NVMeOFConnector, '_execute')
@mock.patch.object(nvmeof.NVMeOFConnector, 'get_volume_paths')
@mock.patch('os_brick.utils.get_device_size')
def test_extend_volume_unreplicated_nvme_fails(self, mock_device_size, mock_paths, mock_exec, mock_lba, mock_property, mock_rescan):
    """nvme command fails, so it rescans, waits, and reads size."""
    dev_path = '/dev/nvme0n1'
    mock_device_size.return_value = 100
    mock_paths.return_value = [dev_path]
    mock_exec.side_effect = putils.ProcessExecutionError()
    self.assertEqual(100, self.connector.extend_volume(connection_properties))
    mock_paths.assert_called_with(mock.ANY)
    self.assertIsInstance(mock_paths.call_args[0][0], nvmeof.NVMeOFConnProps)
    mock_exec.assert_called_once_with('nvme', 'id-ns', '-ojson', dev_path, run_as_root=True, root_helper=self.connector._root_helper)
    mock_lba.assert_not_called()
    mock_property.assert_not_called()
    mock_rescan.assert_called_once_with('nvme0')
    mock_device_size.assert_called_with(self.connector, '/dev/nvme0n1')