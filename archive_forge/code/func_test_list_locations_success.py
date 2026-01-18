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
def test_list_locations_success(self):
    locations = self.driver.list_locations()
    self.assertTrue(len(locations) >= 1)
    location = locations[0]
    self.assertEqual(location.id, 'par1')
    self.assertEqual(location.name, 'Paris 1')