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
def test_ex_delete_anti_affinity_rule_FAIL(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    rule = self.driver.ex_list_anti_affinity_rules(network_domain=net_domain)[0]
    DimensionDataMockHttp.type = 'FAIL'
    with self.assertRaises(DimensionDataAPIException):
        self.driver.ex_delete_anti_affinity_rule(rule)