from unittest import mock
import ddt
from oslo_utils import units
from oslo_vmware.objects import datastore
from oslo_vmware import vim_util
from os_brick import exception
from os_brick.initiator.connectors import vmware
from os_brick.tests.initiator import test_connector
@mock.patch.object(VMDK_CONNECTOR, '_create_spec_for_disk_remove')
@mock.patch.object(VMDK_CONNECTOR, '_reconfigure_backing')
def test_detach_disk_from_backing(self, reconfigure_backing, create_spec):
    disk_spec = mock.sentinel.disk_spec
    create_spec.return_value = disk_spec
    reconfig_spec = mock.Mock()
    session = mock.Mock()
    session.vim.client.factory.create.return_value = reconfig_spec
    backing = mock.sentinel.backing
    disk_device = mock.sentinel.disk_device
    self._connector._detach_disk_from_backing(session, backing, disk_device)
    create_spec.assert_called_once_with(session, disk_device)
    session.vim.client.factory.create.assert_called_once_with('ns0:VirtualMachineConfigSpec')
    self.assertEqual([disk_spec], reconfig_spec.deviceChange)
    reconfigure_backing.assert_called_once_with(session, backing, reconfig_spec)