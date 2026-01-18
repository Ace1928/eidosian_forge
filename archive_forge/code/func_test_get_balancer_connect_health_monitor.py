import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_connect_health_monitor(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    balancer_health_monitor = balancer.extra['healthMonitor']
    self.assertEqual(balancer_health_monitor.type, 'CONNECT')
    self.assertEqual(balancer_health_monitor.delay, 10)
    self.assertEqual(balancer_health_monitor.timeout, 5)
    self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 2)