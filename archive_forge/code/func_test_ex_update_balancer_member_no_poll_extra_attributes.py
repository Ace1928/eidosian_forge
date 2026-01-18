import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_update_balancer_member_no_poll_extra_attributes(self):
    balancer = self.driver.get_balancer(balancer_id='8290')
    members = self.driver.balancer_list_members(balancer)
    first_member = members[0]
    resp = self.driver.ex_balancer_update_member_no_poll(balancer, first_member, condition=MemberCondition.ENABLED, weight=12)
    self.assertTrue(resp)