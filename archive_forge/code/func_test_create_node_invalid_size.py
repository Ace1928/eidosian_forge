import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.compute.base import NodeImage
from libcloud.test.secrets import SCALEWAY_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.scaleway import ScalewayNodeDriver
def test_create_node_invalid_size(self):
    image = NodeImage(id='01234567-89ab-cdef-fedc-ba9876543210', name=None, driver=self.driver)
    size = self.driver.list_sizes()[0]
    location = self.driver.list_locations()[0]
    ScalewayMockHttp.type = 'INVALID_IMAGE'
    expected_msg = '" not found'
    assertRaisesRegex(self, Exception, expected_msg, self.driver.create_node, name='test', size=size, image=image, region=location)