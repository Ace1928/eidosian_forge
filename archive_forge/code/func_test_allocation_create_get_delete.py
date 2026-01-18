import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_create_get_delete(self):
    allocation = self.create_allocation(resource_class=self.resource_class)
    self.assertEqual('allocating', allocation.state)
    self.assertIsNone(allocation.node_id)
    self.assertIsNone(allocation.last_error)
    loaded = self.conn.baremetal.wait_for_allocation(allocation)
    self.assertEqual(loaded.id, allocation.id)
    self.assertEqual('active', allocation.state)
    self.assertEqual(self.node.id, allocation.node_id)
    self.assertIsNone(allocation.last_error)
    with_fields = self.conn.baremetal.get_allocation(allocation.id, fields=['uuid', 'node_uuid'])
    self.assertEqual(allocation.id, with_fields.id)
    self.assertIsNone(with_fields.state)
    node = self.conn.baremetal.get_node(self.node.id)
    self.assertEqual(allocation.id, node.allocation_id)
    self.conn.baremetal.delete_allocation(allocation, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_allocation, allocation.id)