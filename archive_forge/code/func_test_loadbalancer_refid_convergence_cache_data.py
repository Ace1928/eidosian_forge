import copy
from unittest import mock
from oslo_config import cfg
from heat.common import exception
from heat.common import template_format
from heat.engine.clients.os import nova
from heat.engine import node_data
from heat.engine.resources.aws.lb import loadbalancer as lb
from heat.engine import rsrc_defn
from heat.tests import common
from heat.tests.openstack.nova import fakes as fakes_nova
from heat.tests import utils
def test_loadbalancer_refid_convergence_cache_data(self):
    cache_data = {'LoadBalancer': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'reference_id': 'LoadBalancer_convg_mock'})}
    rsrc = self.setup_loadbalancer(cache_data=cache_data)
    self.assertEqual('LoadBalancer_convg_mock', self.stack.defn[rsrc.name].FnGetRefId())