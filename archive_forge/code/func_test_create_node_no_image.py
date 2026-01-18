import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError, InvalidCredsError
from libcloud.compute.base import Node, NodeImage, NodeState, StorageVolume
from libcloud.test.compute import TestCaseMixin
from libcloud.common.linode import LinodeDisk, LinodeIPAddress, LinodeExceptionV4
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.linode import LinodeNodeDriver, LinodeNodeDriverV4
def test_create_node_no_image(self):
    size = self.driver.list_sizes()[0]
    location = self.driver.list_locations()[0]
    LinodeMockHttpV4.type = 'NO_IMAGE'
    node = self.driver.create_node(location, size, name='TestNode', ex_tags=['testing123'])
    self.assertIsNone(node.image)
    self.assertEqual(node.name, 'TestNode')
    self.assertFalse(node.extra['backups']['enabled'])
    self.assertEqual(node.extra['tags'], ['testing123'])
    self.assertEqual(len(node.private_ips), 0)