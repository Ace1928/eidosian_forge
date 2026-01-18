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
def test_ex_edit_firewall_rule_invalid_position_relative_rule_fail(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rule = self.driver.ex_get_firewall_rule(net, 'd0a20f59-77b9-4f28-a63b-e58496b73a6c')
    relative_rule = self.driver.ex_list_firewall_rules(network_domain=net)[-1]
    with self.assertRaises(ValueError):
        self.driver.ex_edit_firewall_rule(rule=rule, position='FIRST', relative_rule_for_position=relative_rule)