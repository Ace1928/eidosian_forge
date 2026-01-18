import os
import sys
import base64
from datetime import datetime
from collections import OrderedDict
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import b, httplib, parse_qs
from libcloud.compute.base import (
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import EC2_PARAMS
from libcloud.compute.types import (
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.ec2 import (
def test_create_encrypted_volume(self):
    location = self.driver.list_locations()[0]
    vol = self.driver.create_volume(10, 'vol', location, ex_encrypted=True, ex_kms_key_id='1234')
    self.assertEqual(10, vol.size)
    self.assertEqual('vol', vol.name)
    self.assertEqual('creating', vol.extra['state'])
    self.assertTrue(isinstance(vol.extra['create_time'], datetime))
    self.assertEqual(True, vol.extra['encrypted'])