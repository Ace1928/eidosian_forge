import sys
import json
import unittest
import libcloud.compute.drivers.equinixmetal
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.compute.base import Node, KeyPair
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.equinixmetal import EquinixMetalNodeDriver
def test_ex_get_node_bandwidth(self):
    node = self.driver.list_nodes('project-id')[0]
    bw = self.driver.ex_get_node_bandwidth(node, 1553194476, 1553198076)
    self.assertTrue(len(bw['bandwidth'][0]['datapoints'][0]) > 0)