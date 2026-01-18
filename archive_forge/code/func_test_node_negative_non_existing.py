import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_negative_non_existing(self):
    uuid = '5c9dcd04-2073-49bc-9618-99ae634d8971'
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.get_node, uuid)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.find_node, uuid, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.delete_node, uuid, ignore_missing=False)
    self.assertRaises(exceptions.ResourceNotFound, self.conn.baremetal.update_node, uuid, name='new-name')
    self.assertIsNone(self.conn.baremetal.find_node(uuid))
    self.assertIsNone(self.conn.baremetal.delete_node(uuid))