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
@mock.patch.object(nvmeof, 'blk_property')
@mock.patch.object(nvmeof.Target, '_get_nvme_devices')
def test_get_device_path_by_initial_devices_multiple_different(self, mock_get_devs, mock_property):
    """There are multiple new devices and they are different."""
    self.target.portals[0].controller = 'nvme0'
    self.target.portals[1].controller = 'nvme1'
    mock_property.side_effect = ('uuid1', 'uuid2')
    mock_get_devs.return_value = ['/dev/nvme0n1', '/dev/nvme0n2', '/dev/nvme1n1', '/dev/nvme1n2']
    self.target.devices_on_start = ['/dev/nvme0n1', '/dev/nvme1n1']
    res = self.target.get_device_path_by_initial_devices()
    mock_get_devs.assert_called_once_with()
    self.assertEqual(2, mock_property.call_count)
    mock_property.assert_has_calls([mock.call('uuid', 'nvme0n2'), mock.call('uuid', 'nvme1n2')], any_order=True)
    self.assertIsNone(res)