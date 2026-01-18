import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, assertRaisesRegex
from libcloud.common.types import InvalidCredsError
from libcloud.compute.types import NodeState
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudsigma import (
def test_ex_create_firewall_policy_no_rules(self):
    CloudSigmaMockHttp.type = 'CREATE_NO_RULES'
    policy = self.driver.ex_create_firewall_policy(name='test policy 1')
    self.assertEqual(policy.name, 'test policy 1')
    self.assertEqual(policy.rules, [])