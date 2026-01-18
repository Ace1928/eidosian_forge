from unittest import mock
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import neutron
from heat.engine.clients.os import nova
from heat.engine.clients.os import sahara
from heat.engine.resources.openstack.sahara import templates as st
from heat.engine import scheduler
from heat.tests import common
from heat.tests import utils
def test_ct_get_live_state(self):
    ct = self._create_ct(self.t)
    resp = mock.MagicMock()
    resp.to_dict.return_value = {'neutron_management_network': 'public', 'description': '', 'cluster_configs': {}, 'created_at': '2016-01-29T11:45:47', 'default_image_id': None, 'updated_at': None, 'plugin_name': 'vanilla', 'shares': None, 'is_default': False, 'is_protected': False, 'use_autoconfig': True, 'anti_affinity': [], 'tenant_id': '221b4f51e9bd4f659845f657a3051a46', 'node_groups': [{'volume_local_to_instance': False, 'availability_zone': None, 'updated_at': None, 'node_group_template_id': '1234', 'volumes_per_node': 0, 'id': '48c356f6-bbe1-4b26-a90a-f3d543c2ea4c', 'security_groups': None, 'shares': None, 'node_configs': {}, 'auto_security_group': False, 'volumes_availability_zone': None, 'volume_mount_prefix': '/volumes/disk', 'floating_ip_pool': None, 'image_id': None, 'volumes_size': 0, 'is_proxy_gateway': False, 'count': 1, 'name': 'test', 'created_at': '2016-01-29T11:45:47', 'volume_type': None, 'node_processes': ['namenode'], 'flavor_id': '2', 'use_autoconfig': True}], 'is_public': False, 'hadoop_version': '2.7.1', 'id': 'c07b8c63-b944-47f9-8588-085547a45c1b', 'name': 'cluster-template-ykokor6auha4'}
    self.ct_mgr.get.return_value = resp
    reality = ct.get_live_state(ct.properties)
    expected = {'neutron_management_network': 'public', 'description': '', 'cluster_configs': {}, 'default_image_id': None, 'plugin_name': 'vanilla', 'shares': None, 'anti_affinity': [], 'node_groups': [{'node_group_template_id': '1234', 'count': 1, 'name': 'test'}], 'hadoop_version': '2.7.1', 'name': 'cluster-template-ykokor6auha4'}
    self.assertEqual(set(expected.keys()), set(reality.keys()))
    expected_node_group = sorted(expected.pop('node_groups'))
    reality_node_group = sorted(reality.pop('node_groups'))
    for i in range(len(expected_node_group)):
        self.assertEqual(expected_node_group[i], reality_node_group[i])
    self.assertEqual(expected, reality)