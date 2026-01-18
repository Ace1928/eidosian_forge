import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_create_balancer_access_rule(self):
    balancer = self.driver.get_balancer(balancer_id='94698')
    rule = RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='0.0.0.0/0')
    rule = self.driver.ex_create_balancer_access_rule(balancer, rule)
    self.assertEqual(2883, rule.id)