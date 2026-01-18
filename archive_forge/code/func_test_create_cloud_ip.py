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
def test_create_cloud_ip(self):
    cip = self.driver.ex_create_cloud_ip()
    self.assertEqual(cip['id'], 'cip-jsjc5')
    self.assertEqual(cip['reverse_dns'], 'cip-109-107-37-234.gb1.brightbox.com')