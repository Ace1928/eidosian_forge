import sys
import unittest
import libcloud.compute.drivers.opennebula
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, NodeSize, NodeImage, NodeState
from libcloud.test.secrets import OPENNEBULA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.opennebula import (
def test_ex_node_action(self):
    """
        Test ex_node_action functionality.
        """
    node = Node(5, None, None, None, None, self.driver)
    ret = self.driver.ex_node_action(node, ACTION.STOP)
    self.assertTrue(ret)