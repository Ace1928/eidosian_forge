import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_fields(self):
    self.create_node()
    result = self.conn.baremetal.nodes(fields=['uuid', 'name', 'instance_id'])
    for item in result:
        self.assertIsNotNone(item.id)
        self.assertIsNone(item.driver)