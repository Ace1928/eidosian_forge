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
def test_ex_authorize_security_group_ingress(self):
    res = self.driver.ex_authorize_security_group_ingress('test_sg', 'udp', '0.0.0.0/0', '0', '65535')
    self.assertEqual(res.get('name'), 'test_sg')
    self.assertTrue('ingressrule' in res)
    rules = res['ingressrule']
    self.assertEqual(len(rules), 1)
    rule = rules[0]
    self.assertEqual(rule['cidr'], '0.0.0.0/0')
    self.assertEqual(rule['endport'], 65535)
    self.assertEqual(rule['protocol'], 'udp')
    self.assertEqual(rule['startport'], 0)