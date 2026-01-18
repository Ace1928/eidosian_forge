import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_create_candidate_nodes(self):
    """Check baremetal allocation create command with candidate nodes.

        Test steps:
        1) Create two nodes.
        2) Create baremetal allocation with specified traits.
        3) Check that allocation successfully created.
        """
    name = data_utils.rand_name('baremetal-allocation')
    node1 = self.node_create(name=name)
    node2 = self.node_create()
    allocation_info = self.allocation_create(params='--candidate-node {0} --candidate-node {1}'.format(node1['name'], node2['uuid']))
    self.assertEqual(allocation_info['state'], 'allocating')
    self.assertIn(node1['uuid'], allocation_info['candidate_nodes'])
    self.assertIn(node2['uuid'], allocation_info['candidate_nodes'])