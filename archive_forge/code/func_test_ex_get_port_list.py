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
def test_ex_get_port_list(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    portlist_id = self.driver.ex_list_portlist(ex_network_domain=net_domain)[0].id
    portlist = self.driver.ex_get_portlist(ex_portlist_id=portlist_id)
    self.assertTrue(isinstance(portlist, DimensionDataPortList))
    self.assertTrue(isinstance(portlist.name, str))
    self.assertTrue(isinstance(portlist.description, str))
    self.assertTrue(isinstance(portlist.state, str))
    self.assertTrue(isinstance(portlist.port_collection, list))
    self.assertTrue(isinstance(portlist.port_collection[0].begin, str))
    self.assertTrue(isinstance(portlist.port_collection[0].end, str))
    self.assertTrue(isinstance(portlist.child_portlist_list, list))
    self.assertTrue(isinstance(portlist.child_portlist_list[0].id, str))
    self.assertTrue(isinstance(portlist.child_portlist_list[0].name, str))
    self.assertTrue(isinstance(portlist.create_time, str))