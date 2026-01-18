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
def test_ex_create_vlan(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    vlan = self.driver.ex_create_vlan(network_domain=net, name='test', private_ipv4_base_address='10.3.4.0', private_ipv4_prefix_size='24', description='test vlan')
    self.assertEqual(vlan.id, '0e56433f-d808-4669-821d-812769517ff8')