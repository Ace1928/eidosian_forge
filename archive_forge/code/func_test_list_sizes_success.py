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
def test_list_sizes_success(self):
    sizes = self.driver.list_sizes()
    self.assertTrue(len(sizes) >= 1)
    size = sizes[0]
    self.assertTrue(size.id is not None)
    self.assertEqual(size.name, 'ARM64-4GB')
    self.assertEqual(size.ram, 4096)
    size = sizes[1]
    self.assertTrue(size.id is not None)
    self.assertEqual(size.name, 'START1-XS')
    self.assertEqual(size.ram, 1024)
    size = sizes[2]
    self.assertTrue(size.id is not None)
    self.assertEqual(size.name, 'X64-120GB')
    self.assertEqual(size.ram, 122880)