import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ALB_PARAMS
from libcloud.loadbalancer.base import Member
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.alb import ApplicationLBDriver
def test_ex_get_balancer_tags(self):
    balancer = self.driver.get_balancer(balancer_id=self.balancer_id)
    self.assertTrue('tags' in balancer.extra, 'No tags dict found in balancer.extra')
    tags = self.driver._ex_get_balancer_tags(balancer)
    self.assertEqual(tags['project'], 'lima')