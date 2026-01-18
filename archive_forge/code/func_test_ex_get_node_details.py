import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import DIGITALOCEAN_v1_PARAMS, DIGITALOCEAN_v2_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.digitalocean import DigitalOcean_v1_Error
from libcloud.compute.drivers.digitalocean import DigitalOceanNodeDriver
def test_ex_get_node_details(self):
    node = self.driver.ex_get_node_details('3164444')
    self.assertEqual(node.name, 'example.com')
    self.assertEqual(node.public_ips, ['36.123.0.123'])
    self.assertEqual(node.extra['image']['id'], 12089443)
    self.assertEqual(node.extra['size_slug'], '8gb')
    self.assertEqual(len(node.extra['tags']), 2)