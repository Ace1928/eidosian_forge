import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_create_get_delete(self):
    node = self.create_node(name='node-name')
    self.assertEqual(node.name, 'node-name')
    self.assertEqual(node.driver, 'fake-hardware')
    self.assertEqual(node.provision_state, 'enroll')
    self.assertFalse(node.is_maintenance)
    for call, ident in [(self.conn.baremetal.get_node, self.node_id), (self.conn.baremetal.get_node, 'node-name'), (self.conn.baremetal.find_node, self.node_id), (self.conn.baremetal.find_node, 'node-name')]:
        found = call(ident)
        self.assertEqual(node.id, found.id)
        self.assertEqual(node.name, found.name)
    with_fields = self.conn.baremetal.get_node('node-name', fields=['uuid', 'driver', 'instance_id'])
    self.assertEqual(node.id, with_fields.id)
    self.assertEqual(node.driver, with_fields.driver)
    self.assertIsNone(with_fields.name)
    self.assertIsNone(with_fields.provision_state)
    nodes = self.conn.baremetal.nodes()
    self.assertIn(node.id, [n.id for n in nodes])
    self.conn.baremetal.delete_node(node, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, self.node_id)