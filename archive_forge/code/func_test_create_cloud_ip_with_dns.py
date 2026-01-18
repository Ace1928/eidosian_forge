import sys
import base64
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import b, httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import BRIGHTBOX_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.brightbox import BrightboxNodeDriver
def test_create_cloud_ip_with_dns(self):
    cip = self.driver.ex_create_cloud_ip('fred.co.uk')
    self.assertEqual(cip['id'], 'cip-jsjc5')
    self.assertEqual(cip['reverse_dns'], 'fred.co.uk')