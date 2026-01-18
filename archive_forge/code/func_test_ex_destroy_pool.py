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
def test_ex_destroy_pool(self):
    response = self.driver.ex_destroy_pool(pool=DimensionDataPool(id='4d360b1f-bc2c-4ab7-9884-1f03ba2768f7', name='test', description='test', status=State.RUNNING, health_monitor_id=None, load_balance_method=None, service_down_action=None, slow_ramp_time=None))
    self.assertTrue(response)