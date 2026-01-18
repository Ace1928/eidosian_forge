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
def test_create_node_ex_userdata(self):
    self.driver.path = '/test/path/userdata'
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    location = self.driver.list_locations()[0]
    CloudStackMockHttp.fixture_tag = 'deploykeyname'
    node = self.driver.create_node(name='test', location=location, image=image, size=size, ex_userdata='foobar')
    self.assertEqual(node.name, 'test')