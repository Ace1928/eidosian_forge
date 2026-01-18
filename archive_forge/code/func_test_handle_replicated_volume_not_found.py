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
@mock.patch.object(nvmeof.NVMeOFConnector, '_is_device_in_raid')
def test_handle_replicated_volume_not_found(self, mock_device_raid):
    mock_device_raid.return_value = False
    conn_props = nvmeof.NVMeOFConnProps(connection_properties)
    conn_props.replica_count = 4
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._handle_replicated_volume, ['/dev/nvme1n1', '/dev/nvme1n2', '/dev/nvme1n3'], conn_props)
    mock_device_raid.assert_any_call('/dev/nvme1n1')
    mock_device_raid.assert_any_call('/dev/nvme1n2')
    mock_device_raid.assert_any_call('/dev/nvme1n3')