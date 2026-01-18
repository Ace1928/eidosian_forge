import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_balancer_http_health_monitor(self):
    balancer = self.driver.get_balancer(balancer_id='94696')
    balancer_health_monitor = balancer.extra['healthMonitor']
    self.assertEqual(balancer_health_monitor.type, 'HTTP')
    self.assertEqual(balancer_health_monitor.delay, 10)
    self.assertEqual(balancer_health_monitor.timeout, 5)
    self.assertEqual(balancer_health_monitor.attempts_before_deactivation, 2)
    self.assertEqual(balancer_health_monitor.path, '/')
    self.assertEqual(balancer_health_monitor.status_regex, '^[234][0-9][0-9]$')
    self.assertEqual(balancer_health_monitor.body_regex, 'Hello World!')