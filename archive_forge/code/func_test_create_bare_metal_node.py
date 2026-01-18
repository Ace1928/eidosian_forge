import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_create_bare_metal_node(self):
    image = self.driver.list_images()[0]
    location = self.driver.list_locations()[0]
    size = self.driver.list_sizes()[-1]
    node = self.driver.create_node(name='test1', image=image, location=location, size=size)
    self.assertEqual(node.name, 'test1')
    self.assertEqual(node.id, '234')
    self.assertTrue(node.extra['is_bare_metal'])
    self.assertEqual(node.extra['cpu_count'], 4)