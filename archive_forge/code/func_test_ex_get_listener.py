import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_listener(self):
    listener = self.driver.ex_get_listener(self.listener_id)
    listener_fields = ('id', 'protocol', 'port', 'action', 'ssl_policy', 'ssl_certificate', '_balancer', '_balancer_arn', '_rules', '_driver')
    for field in listener_fields:
        self.assertTrue(field in listener.__dict__, 'Field [%s] is missing in ALBListener object' % field)
    self.assertEqual(listener.id, self.listener_id)