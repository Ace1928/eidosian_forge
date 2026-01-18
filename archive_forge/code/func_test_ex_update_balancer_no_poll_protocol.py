import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_no_poll_protocol(self):
    balancer = LoadBalancer(id='3130', name='LB_update', state='PENDING_UPDATE', ip='10.34.4.3', port=80, driver=self.driver)
    action_succeeded = self.driver.ex_update_balancer_no_poll(balancer, protocol='HTTPS')
    self.assertTrue(action_succeeded)