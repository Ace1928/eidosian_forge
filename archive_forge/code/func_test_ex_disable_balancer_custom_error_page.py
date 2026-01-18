import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_disable_balancer_custom_error_page(self):
    fixtures = LoadBalancerFileFixtures('rackspace')
    error_page_fixture = json.loads(fixtures.load('error_page_default.json'))
    default_error_page = error_page_fixture['errorpage']['content']
    balancer = self.driver.get_balancer(balancer_id='94695')
    balancer = self.driver.ex_disable_balancer_custom_error_page(balancer)
    error_page_content = self.driver.ex_get_balancer_error_page(balancer)
    self.assertEqual(default_error_page, error_page_content)