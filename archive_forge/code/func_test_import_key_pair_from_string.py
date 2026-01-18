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
def test_import_key_pair_from_string(self):
    result = self.driver.import_key_pair_from_string(name='example', key_material='ssh-rsa AAAAB3NzaC1yc2EAAAADAQABAAAAQQDGk5')
    self.assertTrue(result)