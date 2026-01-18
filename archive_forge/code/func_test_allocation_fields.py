import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_fields(self):
    self.create_allocation(resource_class=self.resource_class)
    result = self.conn.baremetal.allocations(fields=['uuid'])
    for item in result:
        self.assertIsNotNone(item.id)
        self.assertIsNone(item.resource_class)