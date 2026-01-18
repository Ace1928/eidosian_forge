import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_balancer_attach_members(self):
    balancer = self.driver.get_balancer(balancer_id='8292')
    members = [Member(None, ip='10.1.0.12', port='80'), Member(None, ip='10.1.0.13', port='80')]
    attached_members = self.driver.ex_balancer_attach_members(balancer, members)
    first_member = attached_members[0]
    second_member = attached_members[1]
    self.assertEqual(first_member.ip, '10.1.0.12')
    self.assertEqual(first_member.port, 80)
    self.assertEqual(second_member.ip, '10.1.0.13')
    self.assertEqual(second_member.port, 80)