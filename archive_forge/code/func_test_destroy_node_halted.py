import sys
import random
import string
import unittest
from libcloud.utils.py3 import httplib
from libcloud.common.gandi import GandiException
from libcloud.test.secrets import GANDI_PARAMS
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.gandi import GandiNodeDriver
from libcloud.test.common.test_gandi import BaseGandiMockHttp
def test_destroy_node_halted(self):
    nodes = self.driver.list_nodes()
    test_node = list(filter(lambda x: x.state == NodeState.TERMINATED, nodes))[0]
    self.assertTrue(self.driver.destroy_node(test_node))