import sys
import unittest
from libcloud.test import MockHttp, LibcloudTestCase
from libcloud.utils.py3 import httplib
from libcloud.common.types import ServiceUnavailableError
from libcloud.compute.base import NodeSize, NodeImage
from libcloud.test.secrets import VULTR_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.vultr import VultrNodeDriver, VultrNodeDriverV1
def test_list_nodes_require_api_key(self):
    self.driver.list_nodes()
    self.assertTrue(self.driver.connection.require_api_key())