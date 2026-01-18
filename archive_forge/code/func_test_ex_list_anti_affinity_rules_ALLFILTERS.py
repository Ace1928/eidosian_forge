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
def test_ex_list_anti_affinity_rules_ALLFILTERS(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    DimensionDataMockHttp.type = 'ALLFILTERS'
    rules = self.driver.ex_list_anti_affinity_rules(network_domain=net_domain, filter_id='FAKE_ID', filter_state='FAKE_STATE')
    self.assertTrue(isinstance(rules, list))
    self.assertEqual(len(rules), 2)
    self.assertTrue(isinstance(rules[0].id, str))
    self.assertTrue(isinstance(rules[0].node_list, list))