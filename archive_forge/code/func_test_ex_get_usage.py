import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_get_usage(self):
    pricing = self.driver.ex_get_usage()
    self.assertTrue('balance' in pricing)
    self.assertTrue('usage' in pricing)