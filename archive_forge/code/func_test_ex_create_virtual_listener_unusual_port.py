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
def test_ex_create_virtual_listener_unusual_port(self):
    listener = self.driver.ex_create_virtual_listener(network_domain_id='12345', name='test', ex_description='test', port=8900, pool=DimensionDataPool(id='1234', name='test', description='test', status=State.RUNNING, health_monitor_id=None, load_balance_method=None, service_down_action=None, slow_ramp_time=None))
    self.assertEqual(listener.id, '8334f461-0df0-42d5-97eb-f4678eb26bea')
    self.assertEqual(listener.name, 'test')