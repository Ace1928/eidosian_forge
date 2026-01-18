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
def test_ex_list_public_ip_blocks(self):
    net = self.driver.ex_get_network_domain('8cdfd607-f429-4df6-9352-162cfc0891be')
    blocks = self.driver.ex_list_public_ip_blocks(net)
    self.assertEqual(blocks[0].base_ip, '168.128.4.18')
    self.assertEqual(blocks[0].size, '2')
    self.assertEqual(blocks[0].id, '9945dc4a-bdce-11e4-8c14-b8ca3a5d9ef8')
    self.assertEqual(blocks[0].location.id, 'NA9')
    self.assertEqual(blocks[0].network_domain.id, net.id)