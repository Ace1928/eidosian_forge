import copy
from unittest import mock
from heat.common import exception
from heat.engine import node_data
from heat.engine.resources.openstack.heat import resource_chain
from heat.engine import rsrc_defn
from heat.objects import service as service_objects
from heat.tests import common
from heat.tests import utils
def test_get_attribute_convg(self):
    cache_data = {'test-chain': node_data.NodeData.from_dict({'uuid': mock.ANY, 'id': mock.ANY, 'action': 'CREATE', 'status': 'COMPLETE', 'attrs': {'refs': ['rsrc1', 'rsrc2']}})}
    stack = utils.parse_stack(TEMPLATE, cache_data=cache_data)
    rsrc = stack.defn['test-chain']
    self.assertEqual(['rsrc1', 'rsrc2'], rsrc.FnGetAtt('refs'))