import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_rule(self):
    rule = self.driver.ex_get_rule(self.rule_id)
    rule_fields = ('id', 'is_default', 'priority', 'conditions', '_listener', '_listener_arn', '_target_group', '_target_group_arn', '_driver')
    for field in rule_fields:
        self.assertTrue(field in rule.__dict__, 'Field [%s] is missing in ALBRule object' % field)
    self.assertEqual(rule.id, self.rule_id)