import os
import sys
import libcloud.security
from libcloud.test import MockHttp, LibcloudTestCase, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import LibcloudError
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState, NodeAuthPassword
from libcloud.compute.types import Provider
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
def test_destroy_node_success_multiple_nodes_in_cloud_service(self):
    node = Node(id='oddkinz1', name='oddkinz1', state=NodeState.RUNNING, public_ips=[], private_ips=[], driver=self.driver)
    result = self.driver.destroy_node(node=node, ex_cloud_service_name='oddkinz2', ex_deployment_slot='Production')
    self.assertTrue(result)