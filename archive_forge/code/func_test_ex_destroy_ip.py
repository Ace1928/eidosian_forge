import sys
import unittest
from datetime import datetime
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeSize
from libcloud.test.secrets import GRIDSCALE_PARAMS
from libcloud.utils.iso8601 import UTC
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gridscale import GridscaleNodeDriver
def test_ex_destroy_ip(self):
    ip = self.driver.ex_list_ips()[0]
    GridscaleMockHttp.type = 'DELETE'
    self.assertTrue(self.driver.ex_destroy_ip(ip))