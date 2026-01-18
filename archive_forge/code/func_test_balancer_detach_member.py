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
def test_balancer_detach_member(self):
    extra = {'pool_id': '4d360b1f-bc2c-4ab7-9884-1f03ba2768f7', 'network_domain_id': '1234'}
    balancer = LoadBalancer(id='234', name='test', state=State.RUNNING, ip='1.2.3.4', port=1234, driver=self.driver, extra=extra)
    member = Member(id='3dd806a2-c2c8-4c0c-9a4f-5219ea9266c0', ip='112.12.2.2', port=80, balancer=balancer, extra=None)
    result = self.driver.balancer_detach_member(balancer, member)
    self.assertEqual(result, True)