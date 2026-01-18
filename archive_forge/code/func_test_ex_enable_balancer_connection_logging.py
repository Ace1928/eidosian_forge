import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_enable_balancer_connection_logging(self):
    balancer = self.driver.get_balancer(balancer_id='94695')
    balancer = self.driver.ex_enable_balancer_connection_logging(balancer)
    self.assertTrue(balancer.extra['connectionLoggingEnabled'])