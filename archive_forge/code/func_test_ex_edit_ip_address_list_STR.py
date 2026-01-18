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
def test_ex_edit_ip_address_list_STR(self):
    ip_address_1 = DimensionDataIpAddress(begin='190.2.2.111')
    ip_address_collection = [ip_address_1]
    child_ip_address_list = DimensionDataChildIpAddressList(id='2221ef78-4059-4bc1-b433-3f6ad698dc41', name='test_child_ip_address_list edited')
    success = self.driver.ex_edit_ip_address_list(ex_ip_address_list='84e34850-595d- 436e-a885-7cd37edb24a4', description='test ip address list', ip_address_collection=ip_address_collection, child_ip_address_lists=child_ip_address_list)
    self.assertTrue(success)