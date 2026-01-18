import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_list_nodes_with_bare_metals(self):
    nodes = self.driver.list_nodes()
    self.assertEqual(len(nodes), 5)
    node = nodes[0]
    self.assertEqual(node.id, '123')
    self.assertEqual(node.name, 'test1')
    self.assertEqual(node.image, '477')
    self.assertEqual(node.size, 'vc2-1c-2gb')
    self.assertEqual(node.extra['vcpu_count'], 1)
    self.assertEqual(node.extra['location'], 'fra')
    self.assertFalse(node.extra['is_bare_metal'])
    self.assertIn('45.76.83.44', node.public_ips)
    node = nodes[-1]
    self.assertEqual(node.id, '234')
    self.assertEqual(node.size, 'vbm-8c-132gb')
    self.assertEqual(node.state, 'pending')
    self.assertEqual(node.extra['cpu_count'], 8)
    self.assertEqual(node.extra['location'], 'mia')
    self.assertTrue(node.extra['is_bare_metal'])
    self.assertEqual(node.extra['mac_address'], 189250955239968)