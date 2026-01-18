import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import senlin
from heat.engine.resources.openstack.senlin import node as sn
from heat.engine import scheduler
from heat.engine import template
from heat.tests import common
from heat.tests import utils
from openstack import exceptions
def test_node_get_live_state(self):
    expected_reality = {'name': 'SenlinNode', 'metadata': {'foo': 'bar'}, 'profile': 'fake_profile_id', 'cluster': 'fake_cluster_id'}
    node = self._create_node()
    reality = node.get_live_state(node.properties)
    self.assertEqual(expected_reality, reality)