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
@mock.patch.object(nvmeof.Target, 'present_portals', new_callable=mock.PropertyMock)
@mock.patch.object(nvmeof.Target, 'live_portals', new_callable=mock.PropertyMock)
def test_get_devices_get_present(self, mock_live, mock_present):
    """Return all devices that are found."""
    portal1 = mock.Mock(**{'get_device.return_value': '/dev/nvme0n1'})
    portal2 = mock.Mock(**{'get_device.return_value': None})
    portal3 = mock.Mock(**{'get_device.return_value': '/dev/nvme1n1'})
    mock_present.return_value = [portal1, portal2, portal3]
    res = self.target.get_devices(only_live=False)
    self.assertIsInstance(res, list)
    self.assertEqual({'/dev/nvme0n1', '/dev/nvme1n1'}, set(res))
    mock_present.assert_called_once_with()
    mock_live.assert_not_called()
    portal1.get_device.assert_called_once_with()
    portal2.get_device.assert_called_once_with()
    portal3.get_device.assert_called_once_with()