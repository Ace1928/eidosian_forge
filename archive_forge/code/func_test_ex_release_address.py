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
def test_ex_release_address(self):
    EC2MockHttp.type = 'all_addresses'
    elastic_ips = self.driver.ex_describe_all_addresses()
    EC2MockHttp.type = ''
    ret = self.driver.ex_release_address(elastic_ips[2])
    self.assertTrue(ret)
    ret = self.driver.ex_release_address(elastic_ips[0], domain='vpc')
    self.assertTrue(ret)
    self.assertRaises(AttributeError, self.driver.ex_release_address, elastic_ips[0], domain='bogus')