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
def test_ex_edit_firewall_rule_destination_ipaddresslist(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rule = self.driver.ex_get_firewall_rule(net, 'd0a20f59-77b9-4f28-a63b-e58496b73a6c')
    rule.destination.address_list_id = '802abc9f-45a7-4efb-9d5a-810082368222'
    rule.destination.any_ip = False
    rule.destination.ip_address = '10.0.0.1'
    rule.destination.ip_prefix_size = 10
    result = self.driver.ex_edit_firewall_rule(rule=rule, position='LAST')
    self.assertTrue(result)