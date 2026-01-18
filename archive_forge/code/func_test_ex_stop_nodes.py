import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.common.vultr import VultrException
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV2
def test_ex_stop_nodes(self):
    nodes = self.driver.list_nodes()
    response = self.driver.ex_stop_nodes(nodes)
    self.assertTrue(response)