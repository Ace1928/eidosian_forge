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
def test_ex_create_ip_address_list_STR(self):
    name = 'Test_IP_Address_List_3'
    description = 'Test Description'
    ip_version = 'IPV4'
    child_ip_address_list_id = '0291ef78-4059-4bc1-b433-3f6ad698dc41'
    net_domain = self.driver.ex_list_network_domains()[0]
    ip_address_1 = DimensionDataIpAddress(begin='190.2.2.100')
    ip_address_2 = DimensionDataIpAddress(begin='190.2.2.106', end='190.2.2.108')
    ip_address_3 = DimensionDataIpAddress(begin='190.2.2.0', prefix_size='24')
    ip_address_collection = [ip_address_1, ip_address_2, ip_address_3]
    success = self.driver.ex_create_ip_address_list(ex_network_domain=net_domain.id, name=name, ip_version=ip_version, description=description, ip_address_collection=ip_address_collection, child_ip_address_list=child_ip_address_list_id)
    self.assertTrue(success)