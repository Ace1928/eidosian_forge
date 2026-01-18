import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_maintenance(self):
    reason = 'Prepating for taking over the world'
    node = self.create_node()
    self.assertFalse(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.set_node_maintenance(node)
    self.assertTrue(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.set_node_maintenance(node, reason)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)
    node = self.conn.baremetal.set_node_maintenance(node)
    self.assertTrue(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.unset_node_maintenance(node)
    self.assertFalse(node.is_maintenance)
    self.assertIsNone(node.maintenance_reason)
    node = self.conn.baremetal.set_node_maintenance(node, reason)
    self.assertTrue(node.is_maintenance)
    self.assertEqual(reason, node.maintenance_reason)