import random
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_allocation_negative_non_existing(self):
    uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_allocation, uuid)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_allocation, uuid, ignore_missing=False)
    self.assertIsNone(self.conn.baremetal.delete_allocation(uuid))