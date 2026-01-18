import os
import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlparse, parse_qsl, assertRaisesRegex
from libcloud.common.types import ProviderError
from libcloud.compute.base import NodeSize, NodeImage, NodeLocation
from libcloud.test.compute import TestCaseMixin
from libcloud.compute.types import (
from libcloud.compute.providers import get_driver
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.compute.drivers.cloudstack import CloudStackNodeDriver, CloudStackAffinityGroupType
def test_ex_list_port_forwarding_rules(self):
    rules = self.driver.ex_list_port_forwarding_rules()
    self.assertEqual(len(rules), 1)
    rule = rules[0]
    self.assertTrue(rule.node)
    self.assertEqual(rule.protocol, 'tcp')
    self.assertEqual(rule.public_port, '33')
    self.assertEqual(rule.public_end_port, '34')
    self.assertEqual(rule.private_port, '33')
    self.assertEqual(rule.private_end_port, '34')
    self.assertEqual(rule.address.address, '1.1.1.116')