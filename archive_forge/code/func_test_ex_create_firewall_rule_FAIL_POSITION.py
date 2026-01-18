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
def test_ex_create_firewall_rule_FAIL_POSITION(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    rules = self.driver.ex_list_firewall_rules(net)
    with self.assertRaises(ValueError):
        self.driver.ex_create_firewall_rule(net, rules[0], 'BEFORE')