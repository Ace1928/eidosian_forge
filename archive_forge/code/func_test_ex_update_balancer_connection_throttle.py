import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_connection_throttle(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    connection_throttle = RackspaceConnectionThrottle(max_connections=200, min_connections=50, max_connection_rate=50, rate_interval_seconds=10)
    balancer = self.driver.ex_update_balancer_connection_throttle(balancer, connection_throttle)
    updated_throttle = balancer.extra['connectionThrottle']
    self.assertEqual(200, updated_throttle.max_connections)
    self.assertEqual(50, updated_throttle.min_connections)
    self.assertEqual(50, updated_throttle.max_connection_rate)
    self.assertEqual(10, updated_throttle.rate_interval_seconds)