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
def test_create_node_ex_ipaddress(self):
    CloudStackMockHttp.fixture_tag = 'deployip'
    size = self.driver.list_sizes()[0]
    image = self.driver.list_images()[0]
    location = self.driver.list_locations()[0]
    ipaddress = '10.1.0.128'
    networks = [nw for nw in self.driver.ex_list_networks() if str(nw.zoneid) == str(location.id)]
    node = self.driver.create_node(name='deployip', location=location, image=image, size=size, networks=networks, ex_ip_address=ipaddress)
    self.assertEqual(node.name, 'deployip')
    self.assertEqual(node.extra['size_id'], size.id)
    self.assertEqual(node.extra['zone_id'], location.id)
    self.assertEqual(node.extra['image_id'], image.id)
    self.assertEqual(node.private_ips[0], ipaddress)