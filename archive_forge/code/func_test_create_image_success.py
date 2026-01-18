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
def test_create_image_success(self):
    node = self.driver.list_nodes()[0]
    ScalewayMockHttp.type = 'POST'
    image = self.driver.create_image(node, 'my_image')
    self.assertEqual(image.name, 'my_image')
    self.assertEqual(image.id, '98bf3ac2-a1f5-471d-8c8f-1b706ab57ef0')
    self.assertEqual(image.extra['arch'], 'arm')