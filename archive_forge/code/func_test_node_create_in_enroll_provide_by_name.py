import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_create_in_enroll_provide_by_name(self):
    name = 'node-%d' % random.randint(0, 1000)
    node = self.create_node(name=name)
    self.node_id = node.id
    self.assertEqual(node.driver, 'fake-hardware')
    self.assertEqual(node.provision_state, 'enroll')
    self.assertIsNone(node.power_state)
    self.assertFalse(node.is_maintenance)
    node = self.conn.baremetal.set_node_provision_state(name, 'manage', wait=True)
    self.assertEqual(node.provision_state, 'manageable')
    node = self.conn.baremetal.set_node_provision_state(name, 'provide', wait=True)
    self.assertEqual(node.provision_state, 'available')