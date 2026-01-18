from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch.object(VMDK_CONNECTOR, '_reconfigure_backing')
def test_attach_disk_to_backing(self, reconfigure_backing):
    reconfig_spec = mock.Mock()
    disk_spec = mock.Mock()
    session = mock.Mock()
    session.vim.client.factory.create.side_effect = [reconfig_spec, disk_spec]
    backing = mock.Mock()
    disk_device = mock.sentinel.disk_device
    self._connector._attach_disk_to_backing(session, backing, disk_device)
    self.assertEqual([disk_spec], reconfig_spec.deviceChange)
    self.assertEqual('add', disk_spec.operation)
    self.assertEqual(disk_device, disk_spec.device)
    reconfigure_backing.assert_called_once_with(session, backing, reconfig_spec)