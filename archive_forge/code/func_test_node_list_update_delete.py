import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_node_list_update_delete(self):
    self.create_node(name='node-name', extra={'foo': 'bar'})
    node = next((n for n in self.conn.baremetal.nodes(details=True, provision_state='enroll', is_maintenance=False, associated=False) if n.name == 'node-name'))
    self.assertEqual(node.extra, {'foo': 'bar'})
    self.conn.baremetal.update_node(node, extra={'foo': 42})
    self.conn.baremetal.delete_node(node, ignore_missing=False)