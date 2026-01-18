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
def test_ex_create_node(self):
    node = self.driver.ex_create_node(network_domain_id='12345', name='test', ip='123.12.32.2', ex_description='', connection_limit=25000, connection_rate_limit=2000)
    self.assertEqual(node.name, 'myProductionNode.1')
    self.assertEqual(node.id, '9e6b496d-5261-4542-91aa-b50c7f569c54')