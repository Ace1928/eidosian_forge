import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_clone_node(self):
    node_to_clone = self.driver.list_nodes()[0]
    cloned_node = self.driver.ex_clone_node(node=node_to_clone, name='test cloned node')
    self.assertEqual(cloned_node.name, 'test cloned node')