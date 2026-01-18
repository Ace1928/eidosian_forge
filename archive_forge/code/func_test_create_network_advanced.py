from openstack import exceptions
from openstack.tests.functional import base
def test_create_network_advanced(self):
    net1 = self.operator_cloud.create_network(name=self.network_name, shared=True, external=True, admin_state_up=False)
    self.assertIn('id', net1)
    self.assertEqual(self.network_name, net1['name'])
    self.assertTrue(net1['router:external'])
    self.assertTrue(net1['shared'])
    self.assertFalse(net1['admin_state_up'])