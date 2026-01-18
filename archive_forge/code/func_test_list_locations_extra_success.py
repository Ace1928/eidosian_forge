import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def test_list_locations_extra_success(self):
    locations = self.driver.list_locations()
    self.assertTrue(len(locations) >= 1)
    extra_keys = ['continent', 'state', 'ddos_protection', 'block_storage', 'regioncode']
    for location in locations:
        self.assertTrue(len(location.extra.keys()) >= 5)
        self.assertTrue(all((item in location.extra.keys() for item in extra_keys)))