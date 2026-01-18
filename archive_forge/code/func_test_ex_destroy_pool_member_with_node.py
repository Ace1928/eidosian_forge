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
def test_ex_destroy_pool_member_with_node(self):
    response = self.driver.ex_destroy_pool_member(member=DimensionDataPoolMember(id='', name='test', status=State.RUNNING, ip='1.2.3.4', port=80, node_id='34de6ed6-46a4-4dae-a753-2f8d3840c6f9'), destroy_node=True)
    self.assertTrue(response)