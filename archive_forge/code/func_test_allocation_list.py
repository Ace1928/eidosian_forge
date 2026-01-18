import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_list(self):
    allocation1 = self.create_allocation(resource_class=self.resource_class)
    allocation2 = self.create_allocation(resource_class=self.resource_class + '-fail')
    self.conn.baremetal.wait_for_allocation(allocation1)
    self.conn.baremetal.wait_for_allocation(allocation2, ignore_error=True)
    allocations = self.conn.baremetal.allocations()
    self.assertEqual({p.id for p in allocations}, {allocation1.id, allocation2.id})
    allocations = self.conn.baremetal.allocations(state='active')
    self.assertEqual([p.id for p in allocations], [allocation1.id])
    allocations = self.conn.baremetal.allocations(node=self.node.id)
    self.assertEqual([p.id for p in allocations], [allocation1.id])
    allocations = self.conn.baremetal.allocations(resource_class=self.resource_class + '-fail')
    self.assertEqual([p.id for p in allocations], [allocation2.id])