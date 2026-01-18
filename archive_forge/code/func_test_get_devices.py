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
@mock.patch.object(nvmeof.Target, 'factory')
def test_get_devices(self, mock_target):
    """Connector get devices gets devices from all its portals."""
    conn_props = {'vol_uuid': VOL_UUID, 'alias': 'raid_alias', 'replica_count': 2, 'volume_replicas': [{'target_nqn': 'nqn1', 'vol_uuid': VOL_UUID1, 'portals': [['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']]}, {'target_nqn': VOL_UUID2, 'vol_uuid': 'uuid2', 'portals': [['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']]}]}
    targets = [mock.Mock(), mock.Mock()]
    targets[0].get_devices.return_value = []
    targets[1].get_devices.return_value = ['/dev/nvme0n1', '/dev/nvme0n2']
    mock_target.side_effect = targets
    conn_props_instance = nvmeof.NVMeOFConnProps(conn_props)
    res = conn_props_instance.get_devices(mock.sentinel.only_live)
    self.assertListEqual(['/dev/nvme0n1', '/dev/nvme0n2'], res)