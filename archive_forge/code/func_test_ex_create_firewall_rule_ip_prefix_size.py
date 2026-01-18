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
def test_ex_create_firewall_rule_ip_prefix_size(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rule = self.driver.ex_list_firewall_rules(net)[0]
    rule.source.address_list_id = None
    rule.source.any_ip = False
    rule.source.ip_address = '10.2.1.1'
    rule.source.ip_prefix_size = '10'
    rule.destination.address_list_id = None
    rule.destination.any_ip = False
    rule.destination.ip_address = '10.0.0.1'
    rule.destination.ip_prefix_size = '20'
    self.driver.ex_create_firewall_rule(net, rule, 'LAST')