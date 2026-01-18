import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_connection_throttle(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    balancer_connection_throttle = balancer.extra['connectionThrottle']
    self.assertEqual(balancer_connection_throttle.min_connections, 50)
    self.assertEqual(balancer_connection_throttle.max_connections, 200)
    self.assertEqual(balancer_connection_throttle.max_connection_rate, 50)
    self.assertEqual(balancer_connection_throttle.rate_interval_seconds, 10)