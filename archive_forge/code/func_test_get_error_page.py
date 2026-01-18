import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_get_error_page(self):
    balancer = self.driver.get_balancer(balancer_id='18940')
    error_page = self.driver.ex_get_balancer_error_page(balancer)
    self.assertTrue('The service is temporarily unavailable' in error_page)