import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_error_page_no_poll(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    content = '<html>Generic Error Page</html>'
    resp = self.driver.ex_update_balancer_error_page_no_poll(balancer, content)
    self.assertTrue(resp)