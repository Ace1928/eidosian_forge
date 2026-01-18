import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_attach_firewall_policy(self):
    policy = self.driver.ex_list_firewall_policies()[0]
    node = self.driver.list_nodes()[0]
    CloudSigmaMockHttp.type = 'ATTACH_POLICY'
    updated_node = self.driver.ex_attach_firewall_policy(policy=policy, node=node)
    nic = updated_node.extra['nics'][0]
    self.assertEqual(nic['firewall_policy']['uuid'], '461dfb8c-e641-43d7-a20e-32e2aa399086')