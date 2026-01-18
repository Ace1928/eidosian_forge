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
@mock.patch.object(nvmeof, 'nvme_basename', return_value='nvme1n2')
@mock.patch.object(nvmeof, 'sysfs_property')
@mock.patch.object(nvmeof.Portal, 'get_all_namespaces_ctrl_paths')
def test_get_device_by_property_not_found(self, mock_paths, mock_property, mock_name):
    """Exhausts devices searching before returning None."""
    mock_paths.return_value = ['/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2']
    mock_property.side_effect = ['uuid1', 'uuid2']
    self.portal.controller = 'nvme0'
    res = self.portal.get_device_by_property('uuid', 'uuid3')
    self.assertIsNone(res)
    mock_paths.assert_called_once_with()
    self.assertEqual(2, mock_property.call_count)
    mock_property.assert_has_calls([mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n1'), mock.call('uuid', '/sys/class/nvme-fabrics/ctl/nvme0/nvme0n2')])
    mock_name.assert_not_called()