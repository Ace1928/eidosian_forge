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
def test_init_old_props(self, mock_target):
    """Test init with old format connection properties."""
    conn_props = {'nqn': 'nqn_value', 'transport_type': 'rdma', 'target_portal': 'portal_value', 'target_port': 'port_value', 'volume_nguid': 'nguid', 'ns_id': 'nsid', 'host_nqn': 'host_nqn_value', 'qos_specs': None, 'access_mode': 'rw', 'encrypted': False, 'cacheable': True, 'discard': True}
    res = nvmeof.NVMeOFConnProps(conn_props, mock.sentinel.find_controllers)
    self.assertFalse(res.is_replicated)
    self.assertIsNone(res.qos_specs)
    self.assertFalse(res.readonly)
    self.assertFalse(res.encrypted)
    self.assertTrue(res.cacheable)
    self.assertTrue(res.discard)
    self.assertIsNone(res.alias)
    self.assertIsNone(res.cinder_volume_id)
    mock_target.assert_called_once_with(source_conn_props=res, find_controllers=mock.sentinel.find_controllers, volume_nguid='nguid', ns_id='nsid', host_nqn='host_nqn_value', portals=[('portal_value', 'port_value', 'rdma')], vol_uuid=None, target_nqn='nqn_value', qos_specs=None, access_mode='rw', encrypted=False, cacheable=True, discard=True)
    self.assertListEqual([mock_target.return_value], res.targets)