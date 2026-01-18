import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_http_health_monitor_with_no_option_body_regex(self):
    balancer = self.driver.get_balancer(balancer_id='94700')
    monitor = RackspaceHTTPHealthMonitor(type='HTTP', delay=10, timeout=5, attempts_before_deactivation=2, path='/', status_regex='^[234][0-9][0-9]$', body_regex='')
    balancer = self.driver.ex_update_balancer_health_monitor(balancer, monitor)
    updated_monitor = balancer.extra['healthMonitor']
    self.assertEqual('HTTP', updated_monitor.type)
    self.assertEqual(10, updated_monitor.delay)
    self.assertEqual(5, updated_monitor.timeout)
    self.assertEqual(2, updated_monitor.attempts_before_deactivation)
    self.assertEqual('/', updated_monitor.path)
    self.assertEqual('^[234][0-9][0-9]$', updated_monitor.status_regex)
    self.assertEqual('', updated_monitor.body_regex)