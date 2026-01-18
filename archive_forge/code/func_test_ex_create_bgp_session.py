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
def test_ex_create_bgp_session(self):
    node = self.driver.list_nodes('project-id')[0]
    session = self.driver.ex_create_bgp_session(node, 'ipv4')
    self.assertEqual(session['status'], 'unknown')