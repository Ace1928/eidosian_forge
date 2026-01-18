import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_tag_resources(self):
    nodes = self.driver.list_nodes()
    tag = self.driver.ex_list_tags()[0]
    updated_tag = self.driver.ex_tag_resources(resources=nodes, tag=tag)
    self.assertEqual(updated_tag.name, 'test tag 3')