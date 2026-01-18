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
def test_list_key_pairs(self):
    keys = self.driver.list_key_pairs()
    self.assertEqual(len(keys), 1)
    self.assertEqual(keys[0].name, 'example')
    self.assertEqual(keys[0].public_key, 'ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQDGk5')
    self.assertEqual(keys[0].fingerprint, 'f5:d1:78:ed:28:72:5f:e1:ac:94:fd:1f:e0:a3:48:6d')