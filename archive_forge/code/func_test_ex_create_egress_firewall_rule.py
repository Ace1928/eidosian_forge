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
def test_ex_create_egress_firewall_rule(self):
    network_id = '874be2ca-20a7-4360-80e9-7356c0018c0b'
    cidr_list = '192.168.0.0/16'
    protocol = 'TCP'
    start_port = 33
    end_port = 34
    rule = self.driver.ex_create_egress_firewall_rule(network_id, cidr_list, protocol, start_port=start_port, end_port=end_port)
    self.assertEqual(rule.network_id, network_id)
    self.assertEqual(rule.cidr_list, cidr_list)
    self.assertEqual(rule.protocol, protocol)
    self.assertIsNone(rule.icmp_code)
    self.assertIsNone(rule.icmp_type)
    self.assertEqual(rule.start_port, start_port)
    self.assertEqual(rule.end_port, end_port)