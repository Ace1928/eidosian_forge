import sys
from types import GeneratorType
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import ET, httplib
from libcloud.utils.xml import findall, findtext, fixxpath
from libcloud.common.types import InvalidCredsError
from libcloud.compute.base import Node, NodeLocation, NodeAuthPassword
from libcloud.test.compute import TestCaseMixin
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.test.file_fixtures import ComputeFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.compute.drivers.dimensiondata import DimensionDataNic
from libcloud.compute.drivers.dimensiondata import DimensionDataNodeDriver as DimensionData
def test_ex_list_firewall_rules(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rules = self.driver.ex_list_firewall_rules(net)
    self.assertEqual(rules[0].id, '756cba02-b0bc-48f4-aea5-9445870b6148')
    self.assertEqual(rules[0].network_domain.id, '8cdfd607-f429-4df6-9352-162cfc0891be')
    self.assertEqual(rules[0].name, 'CCDEFAULT.BlockOutboundMailIPv4')
    self.assertEqual(rules[0].action, 'DROP')
    self.assertEqual(rules[0].ip_version, 'IPV4')
    self.assertEqual(rules[0].protocol, 'TCP')
    self.assertEqual(rules[0].source.ip_address, 'ANY')
    self.assertTrue(rules[0].source.any_ip)
    self.assertTrue(rules[0].destination.any_ip)