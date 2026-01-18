import random
import uuid
from openstack import exceptions
from openstack.tests.functional.baremetal import base
def test_add_remove_node_trait(self):
    node = self.conn.baremetal.get_node(self.node)
    self.assertEqual([], node.traits)
    self.conn.baremetal.add_node_trait(self.node, 'CUSTOM_FAKE')
    self.assertEqual(['CUSTOM_FAKE'], self.node.traits)
    node = self.conn.baremetal.get_node(self.node)
    self.assertEqual(['CUSTOM_FAKE'], node.traits)
    self.conn.baremetal.add_node_trait(self.node, 'CUSTOM_REAL')
    self.assertEqual(sorted(['CUSTOM_FAKE', 'CUSTOM_REAL']), sorted(self.node.traits))
    node = self.conn.baremetal.get_node(self.node)
    self.assertEqual(sorted(['CUSTOM_FAKE', 'CUSTOM_REAL']), sorted(node.traits))
    self.conn.baremetal.remove_node_trait(node, 'CUSTOM_FAKE', ignore_missing=False)
    self.assertEqual(['CUSTOM_REAL'], self.node.traits)
    node = self.conn.baremetal.get_node(self.node)
    self.assertEqual(['CUSTOM_REAL'], node.traits)