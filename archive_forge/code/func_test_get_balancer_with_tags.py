import sys
import unittest
from libcloud.test import MockHttp
from libcloud.utils.py3 import httplib
from libcloud.test.secrets import LB_ELB_PARAMS
from libcloud.loadbalancer.base import Member, Algorithm
from libcloud.loadbalancer.types import State
from libcloud.test.file_fixtures import LoadBalancerFileFixtures
from libcloud.loadbalancer.drivers.elb import ElasticLBDriver
def test_get_balancer_with_tags(self):
    balancer = self.driver.get_balancer(balancer_id='tests', ex_fetch_tags=True)
    self.assertEqual(balancer.id, 'tests')
    self.assertEqual(balancer.name, 'tests')
    self.assertTrue('tags' in balancer.extra, 'No tags dict found in balancer.extra')
    self.assertEqual(balancer.extra['tags']['project'], 'lima')