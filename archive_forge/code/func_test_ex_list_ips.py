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
def test_ex_list_ips(self):
    ip = self.driver.ex_list_ips()[0]
    self.assertEqual(ip.id, '56b8d161-325b-4fd4')