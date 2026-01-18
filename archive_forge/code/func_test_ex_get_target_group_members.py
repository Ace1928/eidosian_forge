import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_target_group_members(self):
    target_group = self.driver.ex_get_target_group(self.target_group_id)
    target_group_members = self.driver._ex_get_target_group_members(target_group)
    self.assertEqual(len(target_group_members), 1)
    self.assertTrue(hasattr(target_group_members[0], 'id'), 'Target group member is missing "id" field')
    self.assertTrue(hasattr(target_group_members[0], 'port'), 'Target group member is missing "port" field')
    self.assertTrue('health' in target_group_members[0].extra, 'Target group member is missing "health" field')