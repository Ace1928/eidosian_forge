import sys
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib
from libcloud.common.types import InvalidCredsError
from libcloud.test.secrets import DIMENSIONDATA_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.common.dimensiondata import (
from libcloud.loadbalancer.drivers.dimensiondata import DimensionDataLBDriver as DimensionData
def test_ex_get_nodes(self):
    nodes = self.driver.ex_get_nodes()
    self.assertEqual(2, len(nodes))
    self.assertEqual(nodes[0].name, 'ProductionNode.1')
    self.assertEqual(nodes[0].id, '34de6ed6-46a4-4dae-a753-2f8d3840c6f9')
    self.assertEqual(nodes[0].ip, '10.10.10.101')