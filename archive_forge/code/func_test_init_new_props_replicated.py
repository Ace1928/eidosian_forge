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
def test_init_new_props_replicated(self, mock_target):
    """Test init with new format connection properties with replicas."""
    conn_props = {'vol_uuid': VOL_UUID_NO_HYPHENS, 'alias': 'raid_alias', 'replica_count': 2, 'volume_replicas': [{'target_nqn': 'nqn1', 'vol_uuid': VOL_UUID1, 'portals': [['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']]}, {'target_nqn': 'nqn2', 'vol_uuid': VOL_UUID2, 'portals': [['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']]}], 'qos_specs': None, 'access_mode': 'ro', 'encrypted': True, 'cacheable': False, 'discard': False}
    targets = [mock.Mock(), mock.Mock()]
    mock_target.side_effect = targets
    res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
    self.assertTrue(res.is_replicated)
    self.assertIsNone(res.qos_specs)
    self.assertTrue(res.readonly)
    self.assertTrue(res.encrypted)
    self.assertFalse(res.cacheable)
    self.assertFalse(res.discard)
    self.assertEqual('raid_alias', res.alias)
    self.assertEqual(VOL_UUID, res.cinder_volume_id)
    self.assertEqual(2, mock_target.call_count)
    call_1 = dict(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, vol_uuid=VOL_UUID1, target_nqn='nqn1', portals=[['portal1', 'port_value', 'RoCEv2'], ['portal2', 'port_value', 'anything']])
    call_2 = dict(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, vol_uuid=VOL_UUID2, target_nqn='nqn2', portals=[['portal4', 'port_value', 'anything'], ['portal3', 'port_value', 'RoCEv2']])
    mock_target.assert_has_calls([mock.call(**call_1), mock.call(**call_2)])
    self.assertListEqual(targets, res.targets)