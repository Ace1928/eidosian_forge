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
def test_ex_list_portlist(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    portlist = self.driver.ex_list_portlist(ex_network_domain=net_domain)
    self.assertTrue(isinstance(portlist, list))
    self.assertEqual(len(portlist), 3)
    self.assertTrue(isinstance(portlist[0].name, str))
    self.assertTrue(isinstance(portlist[0].description, str))
    self.assertTrue(isinstance(portlist[0].state, str))
    self.assertTrue(isinstance(portlist[0].port_collection, list))
    self.assertTrue(isinstance(portlist[0].port_collection[0].begin, str))
    self.assertTrue(isinstance(portlist[0].port_collection[0].end, str))
    self.assertTrue(isinstance(portlist[0].child_portlist_list, list))
    self.assertTrue(isinstance(portlist[0].child_portlist_list[0].id, str))
    self.assertTrue(isinstance(portlist[0].child_portlist_list[0].name, str))
    self.assertTrue(isinstance(portlist[0].create_time, str))