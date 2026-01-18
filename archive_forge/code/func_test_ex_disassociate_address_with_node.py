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
def test_ex_disassociate_address_with_node(self):
    node = self.driver.list_nodes('project-id')[0]
    assignments = self.driver.ex_list_ip_assignments_for_node(node)
    for ip_assignment in assignments['ip_addresses']:
        if ip_assignment['gateway'] == '147.75.40.2':
            self.driver.ex_disassociate_address(ip_assignment['id'])
            break