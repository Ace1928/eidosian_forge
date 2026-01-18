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
def test_ngt_create(self):
    self._create_ngt(self.t)
    args = {'name': 'node-group-template', 'plugin_name': 'vanilla', 'hadoop_version': '2.3.0', 'flavor_id': 'someflavorid', 'description': '', 'volumes_per_node': 0, 'volumes_size': None, 'volume_type': 'lvm', 'security_groups': None, 'auto_security_group': None, 'availability_zone': None, 'volumes_availability_zone': None, 'node_processes': ['namenode', 'jobtracker'], 'floating_ip_pool': 'some_pool_id', 'node_configs': None, 'image_id': None, 'is_proxy_gateway': True, 'volume_local_to_instance': None, 'use_autoconfig': None, 'shares': [{'id': 'e45eaabf-9300-42e2-b6eb-9ebc92081f46', 'access_level': 'ro', 'path': None}]}
    self.ngt_mgr.create.assert_called_once_with(**args)