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
def test_ex_detach_interface(self):
    ifaces = self.driver.ex_list_interfaces()
    nodes = self.driver.list_nodes()
    res = self.driver.ex_node_detach_interface(nodes[0], ifaces[0])
    self.assertTrue(res)