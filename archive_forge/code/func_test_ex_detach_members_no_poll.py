import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_detach_members_no_poll(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    members = balancer.list_members()
    ret = self.driver.ex_balancer_detach_members_no_poll(balancer, members)
    self.assertTrue(ret)