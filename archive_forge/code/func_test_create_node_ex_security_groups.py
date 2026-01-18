import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_create_node_ex_security_groups(self):
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    location = self.driver.list_locations()[0]
    sg = [sg['name'] for sg in self.driver.ex_list_security_groups()]
    CloudStackMockHttp.fixture_tag = 'deploysecuritygroup'
    node = self.driver.create_node(name='test', location=location, image=image, size=size, ex_security_groups=sg)
    self.assertEqual(node.name, 'test')
    self.assertEqual(node.extra['security_group'], sg)
    self.assertEqual(node.id, 'fc4fd31a-16d3-49db-814a-56b39b9ef986')