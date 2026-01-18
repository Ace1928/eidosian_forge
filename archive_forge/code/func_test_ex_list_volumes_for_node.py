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
def test_ex_list_volumes_for_node(self):
    node = self.driver.list_nodes()[0]
    volumes = self.driver.ex_list_volumes_for_node(node=node)
    self.assertEqual(len(volumes), 1)
    self.assertEqual(volumes[0].size, 50)