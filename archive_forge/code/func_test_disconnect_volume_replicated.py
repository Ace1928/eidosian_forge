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
@mock.patch.object(nvmeof.Target, 'set_portals_controllers', mock.Mock())
@mock.patch('os_brick.initiator.linuxscsi.LinuxSCSI.flush_device_io')
@mock.patch.object(nvmeof.NVMeOFConnector, 'get_volume_paths')
@mock.patch.object(nvmeof.NVMeOFConnector, 'end_raid')
@mock.patch('os.path.exists', return_value=True)
def test_disconnect_volume_replicated(self, mock_exists, mock_end_raid, mock_get_paths, mock_flush):
    """Disconnect a raid."""
    raid_path = '/dev/md/md1'
    mock_get_paths.return_value = [raid_path]
    self.connector.disconnect_volume(connection_properties, mock.sentinel.device_info, ignore_errors=True)
    mock_get_paths.assert_called_once_with(mock.ANY, mock.sentinel.device_info)
    self.assertIsInstance(mock_get_paths.call_args[0][0], nvmeof.NVMeOFConnProps)
    mock_exists.assert_called_once_with(raid_path)
    mock_end_raid.assert_called_with(raid_path)
    mock_flush.assert_not_called()