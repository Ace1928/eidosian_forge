import sys
import datetime
from libcloud.test import MockHttp, unittest
from libcloud.utils.py3 import httplib, urlencode
from libcloud.common.types import LibcloudError
from libcloud.loadbalancer.base import Member, Algorithm, LoadBalancer
from libcloud.loadbalancer.types import MemberCondition
from libcloud.test.file_fixtures import OpenStackFixtures, LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.rackspace import (
def test_ex_create_balancer_access_rules_no_poll(self):
    balancer = self.driver.get_balancer(balancer_id='94699')
    rules = [RackspaceAccessRule(rule_type=RackspaceAccessRuleType.ALLOW, address='2001:4801:7901::6/64'), RackspaceAccessRule(rule_type=RackspaceAccessRuleType.DENY, address='8.8.8.8/0')]
    resp = self.driver.ex_create_balancer_access_rules_no_poll(balancer, rules)
    self.assertTrue(resp)