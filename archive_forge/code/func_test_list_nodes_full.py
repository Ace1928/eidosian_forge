import sys
import json
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.compute import providers
from libcloud.utils.py3 import httplib
from libcloud.compute.base import NodeImage, NodeLocation, NodeAuthSSHKey
from libcloud.test.secrets import KAMATERA_PARAMS
from libcloud.compute.types import Provider, NodeState
from libcloud.common.exceptions import BaseHTTPError
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.kamatera import KamateraNodeDriver
def test_list_nodes_full(self):
    nodes = self.driver.list_nodes(ex_full_details=True)
    self.assertTrue(len(nodes) >= 1)
    node = nodes[0]
    self.assertEqual(node.name, 'my-server')
    self.assertEqual(node.state, NodeState.RUNNING)
    self.assertTrue(len(node.public_ips) > 0)
    self.assertTrue(len(node.private_ips) > 0)
    self.assertEqual(node.driver, self.driver)