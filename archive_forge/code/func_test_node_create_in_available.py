import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_create_in_available(self):
    node = self.create_node(name='node-name', provision_state='available')
    self.assertEqual(node.name, 'node-name')
    self.assertEqual(node.driver, 'fake-hardware')
    self.assertEqual(node.provision_state, 'available')
    self.conn.baremetal.delete_node(node, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, self.node_id)