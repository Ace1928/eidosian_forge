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
def test_ex_get_ip_address_list(self):
    net_domain = self.driver.ex_list_network_domains()[0]
    DimensionDataMockHttp.type = 'FILTERBYNAME'
    ip_list = self.driver.ex_get_ip_address_list(ex_network_domain=net_domain.id, ex_ip_address_list_name='Test_IP_Address_List_3')
    self.assertTrue(isinstance(ip_list, list))
    self.assertEqual(len(ip_list), 1)
    self.assertTrue(isinstance(ip_list[0].name, str))
    self.assertTrue(isinstance(ip_list[0].description, str))
    self.assertTrue(isinstance(ip_list[0].ip_version, str))
    self.assertTrue(isinstance(ip_list[0].state, str))
    self.assertTrue(isinstance(ip_list[0].create_time, str))
    ips = ip_list[0].ip_address_collection
    self.assertEqual(len(ips), 3)
    self.assertTrue(isinstance(ips[0].begin, str))
    self.assertTrue(isinstance(ips[0].prefix_size, str))
    self.assertTrue(isinstance(ips[2].end, str))