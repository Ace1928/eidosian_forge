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
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_raid_device')
@mock.patch.object(nvmeof.NVMeOFConnProps, 'get_devices')
def test_get_volume_paths_unreplicated(self, mock_get_devs, mock_is_raid):
    """Search for device from unreplicated connection properties."""
    mock_get_devs.return_value = ['/dev/nvme0n1']
    conn_props = nvmeof.NVMeOFConnProps(volume_replicas[0])
    res = self.connector.get_volume_paths(conn_props, None)
    self.assertEqual(mock_get_devs.return_value, res)
    mock_is_raid.assert_not_called()
    mock_get_devs.assert_called_once_with()