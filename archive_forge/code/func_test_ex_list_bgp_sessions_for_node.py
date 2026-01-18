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
def test_ex_list_bgp_sessions_for_node(self):
    sessions = self.driver.ex_list_bgp_sessions_for_node(self.driver.list_nodes()[0])
    self.assertEqual(sessions['bgp_sessions'][0]['status'], 'down')